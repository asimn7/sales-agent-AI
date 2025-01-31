import os
import wave
import warnings
import time
import torch
import re
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pyaudio
from dotenv import load_dotenv
import whisper
from openai import OpenAI
import soundfile as sf
from kokoro_onnx import Kokoro
from pynput import keyboard
import json
from datetime import datetime
from time import perf_counter
import onnxruntime as ort
import textwrap
import random

load_dotenv()
warnings.filterwarnings("ignore")

# ---------------------------
# Configuration
# ---------------------------

client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))
GPT_MODEL = "gpt-3.5-turbo"

RECORD_SECONDS = 5
AUDIO_FILENAME = "input.wav"
RESPONSE_AUDIO_FILENAME = "response.wav"
WHISPER_MODEL_NAME = "base"
whisper_model = None

GREETING_MESSAGE = "Welcome to SuperTruck AI!,  I am Alex, your sales representative. How may I assist you today?"

COMPANY_CONTEXT = """
IMPORTANT: avoid using certainly, absolutely etc before starting your response.
REPLY ONLY IN ENGLISH

Your Name is Alex.  
You are a female sales agent who works for SuperTruck AI. 
You are a logistics expert, especially in the field of trucking in the USA. You know all about load booking, dispatching, invoicing, bylaws, and regulations of the trucking industry.  
SuperTruck AI is an AI-based software for the trucking industry, specifically carriers.  

This trucking management app or software helps carriers with load dispatching, invoicing, accounting, IFTA filing, and carrier optimization by increasing profits and reducing trucking costs.  

Your objective as a sales agent is to make callers aware of the services SuperTruck AI offers. Create a trustworthy relationship with the prospect by answering their queries with charm and confidence.  
Your goal is to convert the prospect into a customer by signing up for the service.  
- If the prospect is unsure how SuperTruck AI works, you can arrange a demo for the customer.  
  - Set an appointment at the time preferred by the customer.  
  - Send the appointment calendar details to the customers email and escalate the same to bipingaire@gmail.com.  
- If the customer is busy, set an internal reminder to follow up with him or her.  
- If the prospect is positive and wants to see how it works, send the signup link to their number.  

Core Objectives:  
- Identify and target potential clients by analyzing shipping needs across industries, generating high-quality sales leads, and understanding unique logistics challenges.  
- Develop a sales strategy focused on personalized logistics solutions, customized proposals, effective negotiations, and closing sales deals.  
- Highlight service offerings such as truckload shipping, expedited transportation, cross-border logistics, specialized freight handling, real-time tracking, and logistics consulting.  
- Build long-term client relationships through strategic partnerships, continuous support, prompt inquiry handling, quick issue resolution, and ensuring customer satisfaction.  
- Use a consulting approach to assess client needs, analyze current operations, recommend optimized strategies, identify cost-saving opportunities, and propose technology-enabled solutions.  

Communication Principles:  
- Maintain a professional, solution-oriented tone.  
- Emphasize proactive problem-solving, transparency, and trustworthiness.  
- Focus on client success in all interactions.  

Interaction Guidelines:  
-Interactive communication.
- Actively listen to client needs and ask clarifying questions.  
- Showcase deep logistics expertise and highlight SuperTruck's unique value proposition.  
- Confirm details positively to reinforce trust (e.g., "Great, I'll note that down!").  

Communication Strategy:  
1. First Call:  
   - Start with a warm greeting and introduction.  
   - Build rapport using open-ended questions.  
   - Collect information naturally through conversation.  
   - Offer value before asking for specific details.  
2. Subsequent Calls:  
   - Personalize greetings with the clients name.  
   - Reference previous conversations when relevant.  
   - Continue gathering information seamlessly.  
   - Maintain a consistent, friendly tone.  

Information Collection Guidelines:  
- Avoid asking for multiple details at once.  
- Phrase requests conversationally:  
  - "By the way, could I get your email to send those details?"  
  - "I want to ensure we have the correct information..."  
- Confirm details positively to build confidence (e.g., "Great, I'll note that down!").  
- Immediately store information as it is received.  

Key Information to Collect:  
1. Shipping Requirements:  
   - Current shipping volume, frequency, and cargo types.  
   - Origin and destination routes.  
   - Special handling needs and current logistics challenges.  
   - Budget constraints.  
2. Operational Insights:  
   - Current transportation methods and logistics providers.  
   - Technology integration capabilities.  
   - Growth projections.  
   
important: Try to keep the respones short and to the point. 

"""

KOKORO_MODEL = "kokoro-v0_19.onnx"
KOKORO_VOICES = "voices.bin"
KOKORO_VOICE = "af_sarah"
INTERRUPT_KEY = 'space'


class ConversationHistory:
    def __init__(self, max_history=5):
        self.history = []
        self.max_history = max_history
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.history_file = f"conversation_history_{self.session_id}.json"

    def add_exchange(self, user_text, ai_response):
        exchange = {
            "user": user_text,
            "assistant": ai_response,
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(exchange)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        self._save_history()

    def get_recent_messages(self):
        messages = []
        for exchange in self.history[-self.max_history:]:
            messages.append({"role": "user", "content": exchange["user"]})
            messages.append(
                {"role": "assistant", "content": exchange["assistant"]})
        return messages

    def _save_history(self):
        """Save conversation history to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"[ERROR] Failed to save history: {e}")

    def clear_history(self):
        """Clear history and remove file"""
        self.history = []
        try:
            if os.path.exists(self.history_file):
                os.remove(self.history_file)
        except Exception as e:
            print(f"[ERROR] Failed to remove history file: {e}")


def setup_gpu():
    """Setup GPU acceleration"""
    if torch.cuda.is_available():
        print("[INFO] GPU detected!")
        # Set up CUDA for PyTorch (Whisper)
        torch.cuda.set_device(0)
        # Set up ONNX Runtime for Kokoro
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        return "cuda", providers
    else:
        print("[INFO] No GPU detected, using CPU")
        return "cpu", ['CPUExecutionProvider']


def load_whisper_model():
    global whisper_model
    print("[INFO] Pre-loading Whisper model with GPU...")
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.set_device(0)
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("[INFO] GPU not available, using CPU")

    whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device=device)
    return whisper_model


def record_audio(filename=AUDIO_FILENAME, record_seconds=RECORD_SECONDS):
    chunk = 1024
    fmt = pyaudio.paInt16
    channels = 1
    rate = 16000

    p = pyaudio.PyAudio()
    stream = p.open(format=fmt,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    print(f"[INFO] Recording for {record_seconds} seconds...")
    frames = []
    for _ in range(int(rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)
    print("[INFO] Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(fmt))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))


def preprocess_text(text):
    """Optimize text for faster TTS processing"""
    text = re.sub(r'[,;:]', '.', text)
    text = re.sub(r'(\w+\s+\w+\s+\w+\s+\w+\s+\w+\.)(\s*\w+)', r'\1\n\2', text)
    text = ' '.join(text.split())
    if len(text) > 200:
        sentences = text.split('.')
        text = '.'.join(sentences[:3]) + '.'
    return text


class FastVoiceSystem:
    def __init__(self):
        print("[INFO] Initializing Fast TTS...")
        self.kokoro = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
        self.cache = {}
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.audio_queue = Queue()
        self.is_processing = False
        self.current_audio = None
        self.current_stream = None
        self.pyaudio_instance = pyaudio.PyAudio()
        self.filler_phrases = [
            "Just to be absolutely sure I understand, let me quickly double-check something.",

            "I'm focusing on this for you, so please give me just a second to pull up the details.",
            "To make sure everything is perfect, let me just confirm a few things.",
            "I'm on it. Just a quick check to ensure I've got all the information.",


            "I'm carefully reviewing everything to be certain. Just a moment while I double-check.",
            "I'm making sure I'm giving you the most accurate information, so let me just verify it.",


            "Hold on just a moment while I cross-reference this information.",
            "I'm just checking one last thing to be completely certain.",

        ]
        self.is_greeting = True
        self._warmup_fillers()

    def _warmup_fillers(self):
        """Pre-generate filler audio"""
        print("[INFO] Warming up filler phrases...")
        self.filler_cache = {}
        for phrase in self.filler_phrases:
            samples, sr = self.kokoro.create(
                phrase,
                voice=KOKORO_VOICE,
                speed=1,  # Even slower speed for fillers
                lang="en-us"
            )
            self.filler_cache[phrase] = (samples, sr)

    def text_to_speech(self, text, cache_only=False):
        if not text.strip():
            return

        start_time = perf_counter()
        try:
            # Check if this is a greeting
            is_greeting = text == GREETING_MESSAGE

            # Check cache first
            if text in self.cache:
                if not cache_only:
                    sf.write(RESPONSE_AUDIO_FILENAME,
                             self.cache[text][0], self.cache[text][1])
                    print(
                        f"[INFO] Used cached TTS ({perf_counter() - start_time:.2f}s)")
                return

            # Start filler thread if not cache_only and not greeting
            filler_thread = None
            if not cache_only and not is_greeting:
                filler_thread = Thread(target=self._play_filler)
                filler_thread.start()

            # Split text into chunks and process in parallel
            chunks = self.split_text_into_chunks(text)
            futures = []

            # Submit all chunks for processing
            for chunk in chunks:
                if chunk.strip():
                    future = self.executor.submit(self._process_chunk, chunk)
                    futures.append(future)

            # Process results as they complete
            all_samples = []
            sample_rate = None

            for future in futures:
                samples, sr = future.result()
                if samples is not None and len(samples) > 0:
                    all_samples.append(samples)
                    if sample_rate is None:
                        sample_rate = sr

            # Cache the complete audio
            if all_samples and sample_rate:
                final_audio = np.concatenate(all_samples)
                self.cache[text] = (final_audio, sample_rate)

            # Stop filler and play actual response
            if not cache_only:
                if filler_thread:
                    self.is_processing = False  # This will stop the filler
                    filler_thread.join()
                self._play_complete_response(final_audio, sample_rate)

        except Exception as e:
            print(f"[ERROR] TTS generation failed: {e}")
            import traceback
            traceback.print_exc()

    def _play_filler(self):
        """Play single filler phrase while processing"""
        self.is_processing = True

        # Pick one random filler
        phrase = random.choice(self.filler_phrases)
        samples, sr = self.filler_cache[phrase]

        # Setup stream if needed
        if self.current_stream is None:
            self._setup_audio_stream(sr)

        # Play filler once
        if self.current_stream and self.is_processing:
            try:
                audio_data = (samples * 32767).astype(np.int16).tobytes()
                # Only play if still processing
                if self.is_processing:
                    self.current_stream.write(audio_data)
            except Exception as e:
                print(f"[ERROR] Filler playback failed: {e}")

    def _play_complete_response(self, samples, sample_rate):
        """Play the complete response"""
        self.is_processing = False

        if self.current_stream:
            self.current_stream.stop_stream()
            self.current_stream.close()
            self.current_stream = None

        self._setup_audio_stream(sample_rate)
        if self.current_stream:
            try:
                audio_data = (samples * 32767).astype(np.int16).tobytes()
                self.current_stream.write(audio_data)
            except Exception as e:
                print(f"[ERROR] Response playback failed: {e}")
            finally:
                self.current_stream.stop_stream()
                self.current_stream.close()
                self.current_stream = None

    def _setup_audio_stream(self, sample_rate):
        """Setup audio stream for real-time playback"""
        try:
            self.current_stream = self.pyaudio_instance.open(
                format=self.pyaudio_instance.get_format_from_width(
                    2),  # 16-bit audio
                channels=1,
                rate=sample_rate,
                output=True,
                frames_per_buffer=4096
            )
        except Exception as e:
            print(f"[ERROR] Failed to setup audio stream: {e}")
            self.current_stream = None

    def _process_chunk(self, text):
        """Process a single chunk of text"""
        if not text.strip():
            return None, None

        if text in self.cache:
            return self.cache[text]

        try:
            # Make sure the text ends with proper punctuation
            if not text[-1] in '.!?':
                text = text + '.'

            samples, sample_rate = self.kokoro.create(
                text,
                voice=KOKORO_VOICE,
                speed=1.2,
                lang="en-us"
            )
            return samples, sample_rate
        except Exception as e:
            print(f"[ERROR] Chunk processing failed: {e}")
            return None, None

    def split_text_into_chunks(self, text, words_per_chunk=30):
        """Split text into chunks based on sentence boundaries"""
        if not text.strip():
            return []

        # First split into sentences
        sentences = re.split('(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words = sentence.split()
            word_count = len(sentence_words)

            # If adding this sentence exceeds the chunk size and we already have content
            if current_word_count + word_count > words_per_chunk and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_word_count = 0

            # Add the sentence to current chunk
            current_chunk.append(sentence)
            current_word_count += word_count

        # Add any remaining content
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def _warmup(self):
        """Minimal warmup"""
        essential_phrases = {
            GREETING_MESSAGE,
            "Could you repeat that?",
            "Let me help you."
        }
        for phrase in essential_phrases:
            self.text_to_speech(phrase, cache_only=True)


def transcribe_audio_locally(filename=AUDIO_FILENAME):
    start_time = perf_counter()
    global whisper_model
    try:
        if whisper_model is None:
            whisper_model = load_whisper_model()

        result = whisper_model.transcribe(filename, fp16=False)
        text = result["text"].strip()
        print(
            f"[INFO] Audio transcribed (took {perf_counter() - start_time:.2f}s)")
        return text
    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}")
        return ""


def chat_with_openai(user_text, conversation_history, is_first_message=False):
    start_time = perf_counter()
    if not user_text.strip():
        return "Could you repeat that?"

    try:
        # Build context from conversation history
        context = COMPANY_CONTEXT
        if conversation_history.history:
            context += "\nPrevious conversation context:\n"
            for exchange in conversation_history.history[-3:]:
                context += f"User: {exchange['user']}\nAssistant: {exchange['assistant']}\n"

        messages = [
            {"role": "system", "content": context}
        ]

        messages.extend(conversation_history.get_recent_messages())
        messages.append({"role": "user", "content": user_text})

        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            max_tokens=500
        )
        ai_response = response.choices[0].message.content
        print(
            f"[INFO] GPT response generated (took {perf_counter() - start_time:.2f}s)")

        conversation_history.add_exchange(user_text, ai_response)
        return ai_response

    except Exception as e:
        print(f"[ERROR] GPT API failed: {e}")
        return "I apologize. Could you repeat that?"


def play_audio(filename=RESPONSE_AUDIO_FILENAME, processor=None):
    try:
        wf = wave.open(filename, 'rb')
    except FileNotFoundError:
        return

    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                    frames_per_buffer=4096)

    chunk_size = 4096
    data = wf.readframes(chunk_size)

    while len(data) > 0:
        if processor and not processor.is_speaking:
            break
        stream.write(data)
        data = wf.readframes(chunk_size)

    stream.stop_stream()
    stream.close()
    p.terminate()
    wf.close()


class AudioProcessor:
    def __init__(self):
        self.audio_queue = Queue()
        self.response_queue = Queue()
        self.is_running = True
        self.is_first_interaction = True
        self.voice_system = FastVoiceSystem()
        self.is_speaking = False
        self.conversation_history = ConversationHistory(max_history=5)

        # Play greeting immediately before starting any threads
        print("\n[INFO] Playing greeting...")
        self.voice_system.text_to_speech(GREETING_MESSAGE)
        self._play_response()

        # Start threads after greeting
        self.process_thread = Thread(target=self._process_audio)
        self.process_thread.daemon = True
        self.process_thread.start()

        self.keyboard_listener = keyboard.Listener(
            on_press=self._handle_interrupt)
        self.keyboard_listener.start()

    def _handle_interrupt(self, key):
        if key == keyboard.Key.space and self.is_speaking:
            print("\n[INFO] User interrupted!")
            self.is_speaking = False
            while not self.audio_queue.empty():
                self.audio_queue.get()
            while not self.response_queue.empty():
                self.response_queue.get()

    def _process_audio(self):
        while self.is_running:
            if not self.audio_queue.empty():
                self.audio_queue.get()
                user_text = transcribe_audio_locally()

                if user_text:
                    print(f"\n[USER SAID]: {user_text}")
                    response = chat_with_openai(
                        user_text,
                        self.conversation_history,
                        is_first_message=False  # Changed to False since greeting is handled separately
                    )
                    self.voice_system.text_to_speech(response)
                    self.response_queue.put(True)

                self.audio_queue.task_done()
            time.sleep(0.05)  # Reduced sleep time

    def _play_response(self):
        self.is_speaking = True
        play_audio(processor=self)
        self.is_speaking = False

    def process(self):
        if not self.is_speaking:
            record_audio()
            self.audio_queue.put(True)
            self.response_queue.get()
            self.response_queue.task_done()
            self._play_response()

    def cleanup(self):
        """Clean up all session data"""
        self.conversation_history.clear_history()
        try:
            if os.path.exists(AUDIO_FILENAME):
                os.remove(AUDIO_FILENAME)
            if os.path.exists(RESPONSE_AUDIO_FILENAME):
                os.remove(RESPONSE_AUDIO_FILENAME)
        except Exception as e:
            print(f"Error cleaning up files: {e}")


def main():
    # Pre-load models before creating AudioProcessor
    print("[INFO] Initializing system...")
    load_whisper_model()

    if not os.getenv("OPEN_API_KEY"):
        print("[ERROR] OpenAI API key not found!")
        return

    print("\nStarting SuperTruck AI Customer Service...")
    processor = AudioProcessor()  # This will play the greeting immediately

    print(f"\nPress {INTERRUPT_KEY.upper()} to interrupt the AI at any time")
    print("Press Enter to speak, Ctrl+C to exit")

    try:
        while True:
            input()  # Just wait for Enter key
            if not processor.is_speaking:
                processor.process()
    except KeyboardInterrupt:
        print("\nExiting...")
        processor.cleanup()
        processor.is_running = False


if __name__ == "__main__":
    main()
