import os
import sys

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
def blockPrint():
	print('dummy block')
	# sys.stderr = open(os.devnull, 'w')
def enablePrint():
	print('dummy enable')
	# sys.stderr = sys.__stderr__
# Add the directory containing the executable to the DLL search path for DirectML.dll
if getattr(sys, 'frozen', False):
    # If we are running in a PyInstaller bundle
    exe_dir = os.path.dirname(sys.executable)
    os.add_dll_directory(exe_dir)
    print(f"Added DLL directory for DirectML: {exe_dir}")

blockPrint()
import gradio as gr
import torch
print('Torch Imported')
import whisperx
print('WhisperX Imported')
import gc
import argparse
import inspect
import time
import json
import subprocess
from pathlib import Path
enablePrint()
from scripts.whisper_model import load_custom_model, LANG_CODES
from typing import Optional, Tuple, Callable
from scripts.config_io import read_config_value, write_config_value
from scripts.utils import *  # noqa: F403
import threading
import queue
import re
import sounddevice as sd
import numpy as np
import pyperclip
import time
from datetime import datetime
import platform
# live audio reciever (addon)
mic_queue = queue.Queue()
stream = None
_stream_lock = threading.Lock()
stop_stream = False
gui_queue = queue.Queue()
time_queue = queue.Queue()

def get_input_devices():
	devices = sd.query_devices()
	input_devices = [d["name"] for d in devices if d["max_input_channels"] > 0]
	return input_devices

def get_process_devices():
	devices = []

	try:
		if platform.system() == "Windows":
			import wmi
			c = wmi.WMI()
			cpu_name = c.Win32_Processor()[0].Name.strip()
		else:
			cpu_name = platform.processor()
		devices.append(("CPU (" + cpu_name + ")", "cpu"))
	except:
		devices.append(("CPU", "cpu"))

	try:
		result = subprocess.run(
			["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
			stdout=subprocess.PIPE, stderr=subprocess.PIPE
		)
		if result.returncode == 0:
			gpus = result.stdout.decode().strip().split("\n")
			for gpu in gpus:
				devices.append((f"NVIDIA {gpu}", "cuda"))
	except:
		pass

	try:
		if platform.system() == "Windows":
			import wmi
			c = wmi.WMI()
			gpus = c.Win32_VideoController()
			for gpu in gpus:
				name = gpu.Name.strip()
				if "AMD" in name or "Radeon" in name:
					devices.append((name, "dml"))
	except:
		pass

	try:
		if platform.system() == "Windows":
			import wmi
			c = wmi.WMI()
			gpus = c.Win32_VideoController()
			for gpu in gpus:
				name = gpu.Name.strip()
				if "Intel" in name:
					devices.append((name, "openvino"))
	except:
		pass
	
	return devices


def resolve_process_device(selected_display_name:str):
	devices = get_process_devices()
	print(devices) #prints out only cpu
	for display, backend in devices:
		if display.strip() == selected_display_name.strip():
			return backend
	return "cpu"


# regular expression filter for scripture references  (addon)
def bible_reference_regex(text):
	pattern = r'\b(?:[1-3] )?(?:Genesis|Exodus|Leviticus|Numbers|Deuteronomy|Joshua|Judges|Ruth|Samuel|Kings|Chronicles|Ezra|Nehemiah|Esther|Job|Psalms|Proverbs|Ecclesiastes|Song of Solomon|Isaiah|Jeremiah|Lamentations|Ezekiel|Daniel|Hosea|Joel|Amos|Obadiah|Jonah|Micah|Nahum|Habakkuk|Zephaniah|Haggai|Zechariah|Malachi|Matthew|Mark|Luke|John|Acts|Romans|Corinthians|Galatians|Ephesians|Philippians|Colossians|Thessalonians|Timothy|Titus|Philemon|Hebrews|James|Peter|Jude|Revelation)\b(?:\s+\d{1,3})?(?::\d{1,3})?' 
	match = re.search(pattern, text, re.IGNORECASE)
	return match.group(0) if match else ""

#audio callback for sound device
def audio_callback(indata, frames, time_info, status):
	if status:
		print("Audio status", status)

	arr = np.asarray(indata)
	if arr.ndim > 1:
		arr = arr.mean(axis=1)
	mic_queue.put(arr.astype(np.float32))
	
def start_input_stream(device_name: str, samplerate: int = 16000, blocksize:int = 2048):
	"""
	Start the sounddevice input stream for the selected device.
	Safe to call multiple times (will no-op if already running).
	"""
	global stream
	with _stream_lock:
		if stream is not None and stream.active:
			return

		device_index = None
		devices = sd.query_devices()
		for i, d in enumerate(devices):
			if d["name"] == device_name and d["max_input_channels"] > 0:
				device_index = i
				break
		if device_index is None:
			raise ValueError(f"Input device '{device_name}' not found or has no input channels")

		try:
			stream = sd.InputStream(
				device=device_index,
				channels=1,
				samplerate=samplerate,
				blocksize=blocksize,
				callback=audio_callback
			)
			stream.start()
		except Exception as e:
			stream = None
			raise e

def stop_input_stream():
	global stream
	with _stream_lock:
		try:
			if stream is not None:
				stream.stop()
				stream.close()
		except Exception:
			pass
	stream = None

# ensure gpu_support has correct value
gpu_support, error = read_config_value("gpu_support")
if gpu_support is False:
	write_config_value("gpu_support", "false")
	gpu_support = "false"
if error or gpu_support not in ("false", "cuda", "rocm", "mps"):
	# Check for Apple Silicon MPS
	if torch.backends.mps.is_available():
		write_config_value("gpu_support", "mps")
	# Check for NVIDIA GPU
	elif sys.platform != "darwin":  # Skip nvidia-smi check on macOS
		try:
			result = subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
			if result.returncode == 0:
				write_config_value("gpu_support", "cuda")
			else:
				result = subprocess.run("lspci | grep -i 'amdgpu'", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
				if result.returncode == 0:
					write_config_value("gpu_support", "rocm")
				else:
					write_config_value("gpu_support", "false")
		except FileNotFoundError:
			write_config_value("gpu_support", "false")
	else:
		write_config_value("gpu_support", "false")

# global variables
ALIGN_LANGS = ["en", "fr", "de", "es", "it", "ja", "zh", "nl", "uk", "pt", "ar", "cs", "ru", "pl", "hu", "fi", "fa", "el", "tr", "da", "he", "vi", "ko", "ur", "te", "hi", "ca", "ml", "no", "nn"]
g_model = None
g_model_a = None
g_model_a_metadata = None
g_params = {}
if getattr(sys, "frozen", False): # running from a PyInstaller bundle
	base = Path(sys._MEIPASS)
else:
	base = Path(__file__).resolve().parent

model_dirs = [
	base / "models"/ "whisperx",
	base /"models" /"whisper_original",
	base / "models" /"custom"
]

for dir_path in model_dirs:
	dir_path.mkdir(parents=True, exist_ok=True)
	print(f"Ensured directory: {dir_path} exists ")
lang_path = base / "configs" / "lang.json"
example_path = base / "examples" / "coffe_break_example.mp3"
with lang_path.open("r", encoding="utf-8") as f:
	LANG_DICT = reformat_lang_dict(json.load(f))
val, error = read_config_value("language")
if error:
	write_config_value("language", "en")
	LANG = "en"
else:
	LANG = val
if LANG not in LANG_DICT:
	LANG = "en"
	print(f"WARNING! Language {LANG} not supported for the interface. Using English instead")
MSG: dict[str, str] = LANG_DICT[LANG]

def release_whisper():
	"""
	Release the whisper model from memory.
	"""
	global g_model, g_params
	del g_model
	if g_params.get("device", None) == "gpu":
		torch.cuda.empty_cache()
	else:
		gc.collect()
	g_model = None
	print(MSG["whisper_released"])

def release_align():
	"""
	Release the alignment model from memory.
	"""
	global g_model_a, g_params
	del g_model_a
	if g_params.get("device", None) == "gpu":
		torch.cuda.empty_cache()
	else:
		gc.collect()
	g_model_a = None
	print(MSG["align_released"])

def release_memory_models():
	"""
	Release both models from memory.
	"""
	global g_model, g_model_a, g_params
	del g_model, g_model_a
	if g_params.get("device", None) == "gpu":
		torch.cuda.empty_cache()
	else:
		gc.collect()
	g_model = None
	g_model_a = None
	print(MSG["both_released"])

def get_args_str(func: Callable) -> list:
	"""
	Get the names of the arguments of a function.
	"""
	return list(inspect.signature(func).parameters)

def get_params(
		func: Callable,
		values: list
) -> dict:
	"""
	Get the parameters of a function as a dictionary.
	"""
	keys = get_args_str(func)
	return {k: values[k] for k in keys}

def same_params(
		params1: dict,
		params2: dict,
		*args
) -> bool:
	"""
	Check if two sets of parameters are the same.
	If args are provided, only check the specified parameters.
	"""
	if args:
		return all(params1.get(arg, None) == params2.get(arg, None) for arg in args)
	else:
		return params1 == params2

def transcribe_whisperx(
		model_name: str,
		audio_path: str,
		micro_audio: tuple,
		device: str,
		batch_size: int,
		compute_type: str,
		language: str,
		chunk_size: int,
		beam_size: int,
		release_memory: bool,
		save_root: Optional[str],
		save_audio: bool,
		save_transcription: bool,
		save_alignments: bool,
		save_in_subfolder: bool,
		preserve_name: bool,
		alignments_format: str
) -> Tuple[str, str, str, str]:
	"""
	Transcribe an audio file using the WhisperX model.
		Returns the transcription and sentence-level alignments.
	"""
	print(MSG["inputs_received"])
	if device == "gpu":
		device = "cuda"
	params = get_params(transcribe_whisperx, locals())
	global g_model, g_params

	if not same_params(params, g_params, "language"):
		print(MSG["lang_changed"])
		release_align()

	if not same_params(params, g_params, "model_name", "device", "compute_type", "beam_size") or g_model is None:
		if g_model is not None:
			print(MSG["params_changed"])
			release_whisper()
		print(MSG["loading_model"])
		blockPrint()
		g_model = whisperx.load_model(model_name, device, compute_type=compute_type, asr_options={"beam_size": beam_size}, download_root="models/whisperx")
		enablePrint()
	g_params = params

	return _transcribe()

def transcribe_custom(
		model_name: str,
		audio_path: str,
		micro_audio: tuple,
		device: str,
		batch_size: int,
		compute_type: str,
		language: str,
		chunk_size: int,
		beam_size: int,
		release_memory: bool,
		save_root: Optional[str],
		save_audio: bool,
		save_transcription: bool,
		save_alignments: bool,
		save_in_subfolder: bool,
		preserve_name: bool,
		alignments_format: str
) -> Tuple[str, str, str, str]:
	"""
	Transcribe an audio file using a custom Whisper model.
		Returns the transcription and sentence-level alignments.
	"""
	print(MSG["inputs_received"])
	if device == "gpu":
		device = "cuda"
	params = get_params(transcribe_custom, locals())
	global g_model, g_params

	if not same_params(params, g_params, "language", "device"):
		print(MSG["lang_changed"])
		release_align()

	if not same_params(params, g_params, "model_name", "device", "compute_type", "beam_size") or g_model is None:
		if g_model is not None:
			print(MSG["params_changed"])
			release_memory_models()
		print(MSG["loading_model"])
		blockPrint()
		g_model = load_custom_model(model_name, device, compute_type=compute_type, beam_size=beam_size, download_root="models/custom")
		enablePrint()
	g_params = params

	return _transcribe()

# starting live audio stream (addon)
def start_mic_stream(mic_component):
	global stop_stream
	stop_stream = False

	def loop():
		while not stop_stream:
			data = mic_component.value
			if data is not None:
				mic_queue.put(data)
			time.sleep(0.1)

	threading.Thread(target=loop, daemon=True).start()


def load_whisper_directml(model_name, download_root=None):
	try:
		import torch_directml
		import whisper
		if download_root:
			os.environ["WHISPER_MODEL_DIR"] = str(download_root)
			print(f"Setting Whisper Mode Dir to: {download_root}")

		dml_device = torch_directml.device()
		print(f"Using DirectMl on device: {dml_device}")

		model = whisper.load_model(model_name, device=dml_device)
		original_transcribe = model.transcribe

		def transcriber_wrapper(audio, batch_size=1, language="en", **kwargs):
			if isinstance(audio, np.ndarray) and audio.ndim==1:
				audio = audio.astype(np.float32)
				result = original_transcribe(audio,language=language)
			else:
				result = original_transcribe(audio,language=language)

			return{"segments": result.get("segments", [])}
		
		model.transcribe = transcriber_wrapper
		return model
	except Exception as e:
		print(f"Somthing went wrong: {e}")
		return None

def continous_transcriber(model_name, secs, processing_backend):
	print(secs)
	global stop_stream, g_model

	device = processing_backend
	print(processing_backend, ' dasd') #prints cpu
	compute_type = 'int8'
	SAMPLERATE = 16000    
	SAMPLES_PER_CHUNK = int(SAMPLERATE * secs)
	print(secs)

	# Load model once
	if g_model is None:
		print('Loading... ',device)

		if device == 'dml':
			dml_model = load_whisper_directml(model_name=model_name, download_root=Path("models/whisper_original"))
			
			if dml_model is not None:
				g_model = dml_model
				print('Succesfully Loaded DirectML model for AMD GPU')
			else:
				print("DirectML failed, falling back to CPU")
				device = "cpu"
				g_model = whisperx.load_model(
					model_name, 
					device, 
					compute_type=compute_type, 
					download_root="models/whisperx")
		else:
			g_model = whisperx.load_model(
				model_name, 
				device, 
				compute_type=compute_type, 
				download_root="models/whisperx")
	buffer = []
	concat_len = 0

	while not stop_stream:
		try:
			audio_piece = mic_queue.get(timeout=0.5)
		except queue.Empty:
			continue

		start_time = time.time()
		# ensure numpy array
		audio_piece = np.asarray(audio_piece, dtype=np.float32)
		buffer.append(audio_piece)
		concat_len += audio_piece.shape[0]
		if concat_len >= SAMPLES_PER_CHUNK:
			audio_data = np.concatenate(buffer, axis=0)[:SAMPLES_PER_CHUNK]
			remaining = np.concatenate(buffer, axis=0)[SAMPLES_PER_CHUNK:]
			buffer = [remaining] if remaining.size > 0 else []
			concat_len = remaining.size if remaining.size > 0 else 0

			try:
				result = g_model.transcribe(audio_data, batch_size=1, language="en")
				text = " ".join([seg["text"].strip() for seg in result.get("segments", [])])
			except Exception as e:
				print("Transcription error:", e)
				continue
			# extract bible ref
			#Disabled regex filter for now
			end_time = time.time()
			ref = text 
			if ref:
				pyperclip.copy(ref)
				gui_queue.put([ref, time.time()])
				time_queue.put(f"{(end_time-start_time):.4f}")
				
processed_durations = []
refs = []
def update_live_ref():
	if not gui_queue.empty():
		ref, entry_time = gui_queue.get() 

		exit_time = time.time()
		duration = exit_time - entry_time

		processed_durations.append(f"[{ref}] lasted {duration:.4f} seconds in the GUI Queue and took {time_queue.get()} seconds to transcribe")
		refs.append(ref)

	return "\n".join(refs) if len(refs) > 0 else "Waiting...", "\n".join(processed_durations) if len(processed_durations) > 0 else "Empty"




def start_live_handler(device_name: str, model_name: str, seconds, selected_proc_display):
	"""
	Called from start button. Starts the mic stream and spawns the transcriber thread.
	Returns a short status string to show (optional).
	"""
	global stop_stream
	if device_name is None:
		return "No input device selected"
	stop_stream = False
	try:
		start_input_stream(device_name)
	except Exception as e:
		return f"Failed to start audio device: {e}"

	print(selected_proc_display, 'value test') #only shows the cpu
	processing_backend = resolve_process_device(selected_proc_display)
	print("Using Backend: ", processing_backend, flush=True)

	thread = threading.Thread(target=continous_transcriber, args=(model_name,seconds,processing_backend), daemon=True)
	thread.start()
	return f"Listening on {processing_backend}"

def stop_live_handler():
	"""
	Called from stop button. Signals the transcriber to stop and stops the input stream.
	"""
	global stop_stream
	stop_stream = True
	stop_input_stream()
	refs.clear()
	processed_durations.clear()		
	return "Stopped", "Stopped and Queue Cleared"



def _transcribe() -> Tuple[str, str, str, str]:
	"""
	Transcribe the audio file using the Whisper model.
	Models and parameters should be loaded and stored globally before calling this function.
		Returns the transcription and sentence-level alignments.
	"""
	global g_model, g_model_a, g_model_a_metadata, g_params
	# Create save folder
	save_dir = None
	temp_dir = os.path.join("temp", str(int(time.time())))  # Use timestamp for temp dir
	os.makedirs(temp_dir, exist_ok=True)
	
	if g_params["save_audio"] or g_params["save_transcription"] or g_params["save_alignments"]:
		if g_params["save_root"] is not None and g_params["save_root"] != "":
			save_root = g_params["save_root"]
		else:
			save_root = "outputs"
		if g_params["save_in_subfolder"]:
			save_dir = create_save_folder(save_root)
		else:
			save_dir = save_root

	try:
		# Load (and save) audio
		audio = load_and_save_audio(g_params["audio_path"], g_params["micro_audio"], g_params["save_audio"], save_dir, g_params["preserve_name"])

		# Transcription
		if g_params["language"] == "auto": 
			language = None
		else:
			language = g_params["language"]
		time_transcribe = time.time()
		print(MSG["starting_transcription"])
		result = g_model.transcribe(audio, batch_size=g_params["batch_size"], language=language, chunk_size=g_params["chunk_size"], print_progress=True)
		if "time" in result.keys():
			time_transcribe = result["time"]
		else:
			time_transcribe = time.time() - time_transcribe
		joined_text = " ".join([segment["text"].strip() for segment in result["segments"]])
		if g_params["save_transcription"]:
			if g_params["preserve_name"]:
				audio_name = os.path.basename(g_params["audio_path"]).split(".")[0]
				save_name = f"{audio_name}_transcription.txt"
			else:
				save_name = "transcription.txt"
			save_transcription_to_txt(joined_text, save_dir, save_name)

		if g_params["release_memory"]:
			release_whisper()


		"""
		Disabling alignment because is not needed for scripture transcribing
		"""
		# Word-level alignment
		# lang_used = result["language"]
		# if lang_used not in ALIGN_LANGS:
		# 	print(MSG["align_lang_not_supported"].format(lang_used))
		# 	lang_used = "en"
		# if g_model_a is None:
		# 	print(MSG["loading_align_model"])
		# 	g_model_a, g_model_a_metadata = whisperx.load_align_model(language_code=lang_used, device=g_params["device"], model_dir="models/alignment")
		# print(MSG["aligning"])
		# time_align = time.time()
		# aligned_result = whisperx.align(result["segments"], g_model_a, g_model_a_metadata, audio, g_params["device"], return_char_alignments=False)
		# time_align = time.time() - time_align
		# if g_params["save_alignments"]:
		# 	align_format = g_params["alignments_format"].lower()
		# 	if g_params["preserve_name"]:
		# 		audio_name = os.path.basename(g_params["audio_path"]).split(".")[0]
		# 		save_name = f"{audio_name}_timestamps." + align_format
		# 	else:
		# 		save_name = "timestamps." + align_format
		# 	if align_format == "json":
		# 		save_alignments_to_json(aligned_result, save_dir, save_name)
		# 	elif align_format == "srt":
		# 		subtitles = alignments2subtitles(aligned_result["segments"], max_line_length=50)
		# 		save_subtitles_to_srt(subtitles, save_dir, save_name)
		# if g_params["release_memory"]:
		# 	release_align()

		"""
		Dummy alignment to avoid removing the alignment UI
		"""
		aligned_result = {"segments": result["segments"]}
		time_align = 0
		print(MSG["done"])
		
		return joined_text, "Alignment disabled", f"{round(time_transcribe, 3)}s", "0s"
	finally:
		# Clean up temp directory
		try:
			import shutil
			if os.path.exists(temp_dir):
				shutil.rmtree(temp_dir)
		except Exception as e:
			print(f"Warning: Could not clean up temp directory: {e}")


# Prepare interface data
whisperx_models = ["large-v3", "large-v2", "large-v1", "medium", "small", "base", "tiny", "medium.en", "small.en", "base.en", "tiny.en"]
custom_models = list_models()
whisperx_langs = ["auto", "en", "es", "fr", "de", "it", "ja", "zh", "nl", "uk", "pt"]
custom_langs = ["auto"] + list(LANG_CODES.keys())

# Read config
gpu_support, error = read_config_value("gpu_support")
if gpu_support in ("cuda", "rocm"):
	device = "gpu"
	device_interactive = True
	device_message = ""
else:
	device = "cpu"
	device_interactive = False
	if gpu_support is None:
		device_message = MSG["select_cpu"]
	else:
		device_message = MSG["gpu_disabled"]

def apply_config(lang: str):
	prev_lang, error = read_config_value("language")
	prev_lang = prev_lang if not error else LANG
	write_config_value("language", lang)
	if lang != prev_lang:
		gr.Info(MSG["settings_updated"])

# Gradio interface
with gr.Blocks(title="Whisper GUI") as demo:
	gr.Markdown(f"""# Whisper GUI
{MSG["gui_description"]}""")
	with gr.Tab("Faster Whisper"):
		with gr.Row():
			with gr.Column():
				model_select = gr.Dropdown(whisperx_models, value="tiny.en", label=MSG["model_select_label"], info=MSG["change_whisper_reload"])
				with gr.Group():
					devices = get_process_devices()
					print(devices, ' jhkhkj') #prints [('CPU (Intel(R) Core(TM) i7-4600U CPU @ 2.10GHz)', 'cpu'), ('Intel(R) HD Graphics Family', 'openvino')]  jhkhkj
					proc_device = gr.Dropdown(choices=[d[0] for d in devices],value=devices[0][0], label='Select Processing Device', allow_custom_value=True, interactive=True)
					input_device = gr.Dropdown(choices=get_input_devices(), value=(get_input_devices()[0] if get_input_devices() else None), label="Select input device", allow_custom_value=True)
					samplerate_box = gr.Number(value=16000, label="Sample rate (Hz)", interactive=False)
					refresh_button = gr.Button("Refresh Input Devices")
					refresh_button.click(lambda: get_input_devices(), outputs=[input_device])
			
					"""
					No longer used as input method Chnaged
					"""
					# submit_button = gr.Button(value=MSG["submit_button"])
			with gr.Column():
				"""
				Transcription and alignment no longer needed
				"""
				# transcription_output = gr.Textbox(label=MSG["transcription_textbox"], lines=15)
				# alignments_output = gr.Textbox(label=MSG["align_textbox"], lines=15)
				# with gr.Row():
				# 	time_transcribe = gr.Textbox(label=MSG["time_transcribe_label"], info=MSG["time_transcribe_info"], lines=1)
				# 	time_align = gr.Textbox(label=MSG["time_align_label"], lines=1)

				#live audio Ui (addon)
				chunk_seconds = gr.Slider(1, 60, step=1, label="Chunk Size Time", info="This is how long the app will wait before it transcribes live audio",interactive=True)
				def updateTimer(x):
					return gr.Timer(float(x), active=True)
				dummy = gr.Textbox(visible=False)
				live_ref_box = gr.Textbox(label="Live Bible Reference", lines=2, max_lines=5)
				time_box = gr.Textbox(label='Queue', lines=2, max_lines=5)
				refresh_button = gr.Button("Refresh Live Reference")
				refresh_button.click(fn=update_live_ref, inputs=[], outputs=[live_ref_box])
				start_live = gr.Button("Start Live Transcription")
				stop_live = gr.Button("Stop Live Transcription")
				release_memory_button = gr.Button(value=MSG["release_memory_button"])
				#Auto updater
				ref_timer = gr.Timer(1)
				ref_timer_2 = gr.Timer(0.1)
				ref_timer_2.tick(fn=update_live_ref, inputs=None, outputs=[live_ref_box, time_box])
					

	with gr.Tab("Custom model"):
		with gr.Row():
			with gr.Column():
				with gr.Group():
					model_select2 = gr.Dropdown(custom_models, value=None, label=MSG["model_select2_label"], allow_custom_value=True, info=MSG["change_whisper_reload"])
				with gr.Group():
					file_upload2 = gr.File(
						label="Upload Audio/Video File",
						file_types=[".mp3", ".wav", ".m4a", ".mp4", ".avi", ".mov", ".mkv", ".webm"],
						type="filepath"
					)
					audio_record2 = gr.Audio(sources=["microphone"], type="numpy", label=MSG["audio_record_label"], visible=False)
					save_audio2 = gr.Checkbox(value=False, label="Save extracted audio", info="Save the audio/extracted audio to the output directory")
				gr.Examples(examples=[str(example_path)], inputs=file_upload2)
				with gr.Accordion(label=MSG["advanced_options"], open=False):
					language_select2 = gr.Dropdown(custom_langs, value = "auto", label="Language", info=MSG["language_select_info"]+MSG["change_align_reload"])
					device_select2 = gr.Radio(["gpu", "cpu"], value = device, label=MSG["device_select_label"], info=device_message+MSG["change_both_reload"], interactive=device_interactive)
					with gr.Group():
						with gr.Row():
							save_transcription2 = gr.Checkbox(value=True, label=MSG["save_transcription_label"])
							save_alignments2 = gr.Checkbox(value=True, label=MSG["save_align_label"])
						save_root2 = gr.Textbox(label=MSG["save_root_label"], placeholder="outputs", lines=1)
						save_in_subfolder2 = gr.Checkbox(value=True, label=MSG["save_subfolder_label"], info=MSG["save_subfolder_info"])
						preserve_name2 = gr.Checkbox(value=False, label=MSG["preserve_name_label"], info=MSG["preserve_name_info"])
						alignments_format2 = gr.Radio(["JSON", "SRT"], value="JSON", label=MSG["align_format_label"], interactive=True)
					gr.Markdown(f"""### {MSG["optimizations"]}""")
					compute_type_select2 = gr.Radio(["float16", "float32"], value = "float16", label=MSG["compute_type_label"], info=MSG["compute_type_info"]+MSG["change_whisper_reload"])
					batch_size_slider2 = gr.Slider(1, 128, value = 1, step=1, label=MSG["batch_size_label"], info=MSG["batch_size_info"])
					chunk_size_slider2 = gr.Slider(1, 80, value = 20, step=1, label=MSG["chunk_size_label"], info=MSG["chunk_size_info"])
					beam_size_slider2 = gr.Slider(1, 100, value = 5, step=1, label=MSG["beam_size_label"], info=MSG["beam_size_info"]+MSG["change_whisper_reload"])
					release_memory_checkbox2 = gr.Checkbox(label=MSG["release_memory_label"], value=True, info=MSG["release_memory_info"])
				submit_button2 = gr.Button(value=MSG["submit_button"])
			with gr.Column():
				transcription_output2 = gr.Textbox(label=MSG["transcription_textbox"], lines=15)
				alignments_output2 = gr.Textbox(label=MSG["align_textbox"], lines=15)
				with gr.Row():
					time_transcribe2 = gr.Textbox(label=MSG["time_transcribe_label"], info=MSG["time_transcribe_info"], lines=1)
					time_align2 = gr.Textbox(label=MSG["time_align_label"], lines=1)
				release_memory_button2 = gr.Button(value=MSG["release_memory_button"])

	with gr.Tab("Settings"):
		lang_select = gr.Dropdown(LANG_DICT.keys(), value=LANG, label=MSG["lang_select_label"], allow_custom_value=True, info=MSG["lang_select_info"])
		apply_button = gr.Button(value=MSG["apply_changes"])
	
	"""
	No longer used since input method has changed
	"""
	# submit_button.click(transcribe_whisperx,
	# 					inputs=[model_select, file_upload, audio_record, device_select, batch_size_slider, compute_type_select, language_select, chunk_size_slider, beam_size_slider, release_memory_checkbox, save_root, save_audio, save_transcription, save_alignments, save_in_subfolder, preserve_name, alignments_format],
	# 					outputs=[transcription_output, alignments_output, time_transcribe, time_align])
	
	# live audio stream button bindings (addon)
	start_live.click(
		fn=start_live_handler,
		inputs=[input_device, model_select, chunk_seconds, proc_device],   
		outputs=[live_ref_box],                             
	)

	stop_live.click(
		fn=stop_live_handler,
		outputs=[live_ref_box]
	)

	chunk_seconds.change(
	fn=updateTimer,
	inputs=[chunk_seconds],
	outputs=[ref_timer]

	)

	demo.load(
		fn=update_live_ref,
		inputs=None,
		js="""
			setInterval(()=>{console.log(200)}, 500)
		"""
	)

	
	submit_button2.click(transcribe_custom,
						inputs=[model_select2, file_upload2, audio_record2, device_select2, batch_size_slider2, compute_type_select2, language_select2, chunk_size_slider2, beam_size_slider2, release_memory_checkbox2, save_root2, save_audio2, save_transcription2, save_alignments2, save_in_subfolder2, preserve_name2, alignments_format2],
						outputs=[transcription_output2, alignments_output2, time_transcribe2, time_align2])
	
	release_memory_button.click(release_memory_models)
	release_memory_button2.click(release_memory_models)

	apply_button.click(apply_config, inputs=[lang_select])
	
	


if __name__ == "__main__":
	# Parse arguments
	parser = argparse.ArgumentParser(description=MSG["argparse_description"])
	parser.add_argument("--autolaunch", action="store_true", default=False, help=MSG["autloaunch_help"])
	parser.add_argument("--share", action="store_true", default=False, help=MSG["share_help"])
	args = parser.parse_args()

	# Launch the interface
	print(MSG["creating_interface"])
	# When running in Docker, we need to bind to 0.0.0.0
	is_docker = os.path.exists('/.dockerenv')
	demo.launch(
		inbrowser=args.autolaunch,
		share=args.share,
		server_name='0.0.0.0' if is_docker else None
	)
