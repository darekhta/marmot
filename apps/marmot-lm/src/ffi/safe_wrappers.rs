//! Safe Rust wrappers around the marmot FFI bindings
//!
//! This module exposes the full marmot API surface. Not all functions are used
//! internally - some are reserved for future features (CPU backend, request
//! cancellation, etc.) or for external callers.

use std::collections::HashMap;
use std::ffi::{c_void, CStr, CString};
use std::path::Path;
use std::ptr;
use std::sync::{Arc, Mutex};

use super::bindings::*;

/// Marmot error type
#[derive(Debug, Clone)]
pub struct MarmotError {
    pub code: i32,
    pub message: String,
}

impl std::fmt::Display for MarmotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MarmotError({}): {}", self.code, self.message)
    }
}

impl std::error::Error for MarmotError {}

pub type Result<T> = std::result::Result<T, MarmotError>;

fn error_code_name(code: marmot_error_t) -> Option<&'static str> {
    match code {
        MARMOT_OK => Some("Success"),
        MARMOT_ERROR_INVALID_ARGUMENT => Some("Invalid argument"),
        MARMOT_ERROR_OUT_OF_MEMORY => Some("Out of memory"),
        MARMOT_ERROR_DEVICE_NOT_AVAILABLE => Some("Device not available"),
        MARMOT_ERROR_BACKEND_INIT_FAILED => Some("Backend init failed"),
        MARMOT_ERROR_INVALID_OPERATION => Some("Invalid operation"),
        MARMOT_ERROR_UNSUPPORTED_DTYPE => Some("Unsupported dtype"),
        MARMOT_ERROR_DIMENSION_MISMATCH => Some("Dimension mismatch"),
        MARMOT_ERROR_NOT_IMPLEMENTED => Some("Not implemented"),
        _ => None,
    }
}

fn check_error(code: marmot_error_t) -> Result<()> {
    if code == MARMOT_OK {
        Ok(())
    } else {
        let raw_message = unsafe {
            let ptr = marmot_error_string(code);
            if ptr.is_null() {
                String::new()
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        let message = if raw_message.is_empty() || raw_message == "Unknown error" {
            error_code_name(code)
                .unwrap_or("Unknown error")
                .to_string()
        } else {
            raw_message
        };
        Err(MarmotError { code, message })
    }
}

fn truncate_c_string(buffer: &mut Vec<u8>, len: usize) {
    if len == 0 {
        buffer.clear();
        return;
    }
    if buffer.len() >= len && buffer[len - 1] == 0 {
        buffer.truncate(len - 1);
    } else {
        buffer.truncate(len);
    }
}

/// Backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Metal,
}

impl Backend {
    fn to_raw(self) -> marmot_backend_type_t {
        match self {
            Backend::Cpu => MARMOT_BACKEND_CPU,
            Backend::Metal => MARMOT_BACKEND_METAL,
        }
    }
}

/// Marmot context - manages device resources
pub struct Context {
    ptr: *mut marmot_context_t,
}

impl Context {
    pub fn new(backend: Backend) -> Result<Self> {
        let ptr = unsafe { marmot_init(backend.to_raw()) };
        if ptr.is_null() {
            let raw_code = unsafe { marmot_get_last_error() };
            let code = if raw_code == MARMOT_OK {
                MARMOT_ERROR_INVALID_OPERATION
            } else {
                raw_code
            };
            let detail = unsafe {
                let ptr = marmot_get_last_error_detail();
                if ptr.is_null() {
                    String::new()
                } else {
                    CStr::from_ptr(ptr).to_string_lossy().into_owned()
                }
            };
            let message = if !detail.is_empty() {
                detail
            } else {
                unsafe {
                    let ptr = marmot_error_string(code);
                    if ptr.is_null() {
                        "Failed to create context".to_string()
                    } else {
                        CStr::from_ptr(ptr).to_string_lossy().into_owned()
                    }
                }
            };
            Err(MarmotError { code, message })
        } else {
            Ok(Self { ptr })
        }
    }

    pub(crate) fn as_ptr(&self) -> *const marmot_context_t {
        self.ptr
    }

    pub(crate) fn backend_type(&self) -> marmot_backend_type_t {
        unsafe { marmot_context_get_backend(self.ptr) }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { marmot_destroy(self.ptr) };
        }
    }
}

unsafe impl Send for Context {}
unsafe impl Sync for Context {}

/// Model information
#[derive(Debug, Clone)]
// Fields populated from C struct; not all consumed by Rust callers yet
#[allow(dead_code)]
pub struct ModelInfo {
    pub architecture: String,
    pub context_length: usize,
    pub n_vocab: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub ff_length: usize,
    pub rope_dimension: usize,
    pub rope_freq_base: f32,
    pub rope_type: marmot_rope_type_t,
    pub rope_scaling_type: marmot_rope_scaling_type_t,
    pub rope_freq_scale: f32,
    pub rope_ext_factor: f32,
    pub rope_attn_factor: f32,
    pub rope_beta_fast: f32,
    pub rope_beta_slow: f32,
    pub rope_orig_ctx_len: u32,
    pub rms_norm_eps: f32,
}

/// Loaded model
pub struct Model {
    ptr: *mut marmot_model_t,
}

impl Model {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str =
            CString::new(path.as_ref().to_string_lossy().as_bytes()).map_err(|_| MarmotError {
                code: MARMOT_ERROR_INVALID_ARGUMENT,
                message: "Invalid path".to_string(),
            })?;

        let mut opts = std::mem::MaybeUninit::<marmot_model_options_t>::uninit();
        unsafe {
            check_error(marmot_model_options_init(opts.as_mut_ptr()))?;
        }

        let mut model_ptr: *mut marmot_model_t = ptr::null_mut();
        unsafe {
            check_error(marmot_model_load_file(
                path_str.as_ptr(),
                opts.as_ptr(),
                &mut model_ptr,
            ))?;
        }

        Ok(Self { ptr: model_ptr })
    }

    pub fn info(&self) -> Result<ModelInfo> {
        let mut info = std::mem::MaybeUninit::<marmot_model_info_t>::uninit();
        unsafe {
            check_error(marmot_model_get_info(self.ptr, info.as_mut_ptr()))?;
            let info = info.assume_init();

            let arch_bytes: Vec<u8> = info
                .architecture
                .iter()
                .take_while(|&&c| c != 0)
                .map(|&c| c as u8)
                .collect();
            let architecture = String::from_utf8_lossy(&arch_bytes).into_owned();

            Ok(ModelInfo {
                architecture,
                context_length: info.context_length,
                n_vocab: info.n_vocab,
                n_embd: info.n_embd,
                n_layer: info.n_layer,
                n_head: info.n_head,
                n_head_kv: info.n_head_kv,
                ff_length: info.ff_length,
                rope_dimension: info.rope_dimension,
                rope_freq_base: info.rope_freq_base,
                rope_type: info.rope_type,
                rope_scaling_type: info.rope_scaling_type,
                rope_freq_scale: info.rope_freq_scale,
                rope_ext_factor: info.rope_ext_factor,
                rope_attn_factor: info.rope_attn_factor,
                rope_beta_fast: info.rope_beta_fast,
                rope_beta_slow: info.rope_beta_slow,
                rope_orig_ctx_len: info.rope_orig_ctx_len,
                rms_norm_eps: info.rms_norm_eps,
            })
        }
    }

    pub(crate) fn as_ptr(&self) -> *const marmot_model_t {
        self.ptr
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { marmot_model_destroy(self.ptr) };
        }
    }
}

unsafe impl Send for Model {}
unsafe impl Sync for Model {}

/// Special token IDs
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    pub bos_id: Option<i32>,
    pub eos_id: Option<i32>,
    pub unk_id: Option<i32>,
    pub pad_id: Option<i32>,
}

/// Tokenizer
pub struct Tokenizer {
    ptr: *mut marmot_tokenizer_t,
    default_add_bos: Option<bool>,
    default_add_eos: Option<bool>,
}

impl Tokenizer {
    pub fn from_gguf<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str =
            CString::new(path.as_ref().to_string_lossy().as_bytes()).map_err(|_| MarmotError {
                code: MARMOT_ERROR_INVALID_ARGUMENT,
                message: "Invalid path".to_string(),
            })?;

        let mut opts = std::mem::MaybeUninit::<marmot_tokenizer_options_t>::uninit();
        unsafe {
            check_error(marmot_tokenizer_options_init(opts.as_mut_ptr()))?;
        }

        let mut tokenizer_ptr: *mut marmot_tokenizer_t = ptr::null_mut();
        unsafe {
            check_error(marmot_tokenizer_create_from_gguf_file(
                path_str.as_ptr(),
                opts.as_ptr(),
                &mut tokenizer_ptr,
            ))?;
        }

        let mut default_add_bos = None;
        let mut default_add_eos = None;
        unsafe {
            let mut behavior = std::mem::MaybeUninit::<marmot_tokenizer_behavior_t>::uninit();
            if marmot_tokenizer_get_behavior(tokenizer_ptr, behavior.as_mut_ptr()) == MARMOT_OK {
                let behavior = behavior.assume_init();
                if behavior.has_add_bos {
                    default_add_bos = Some(behavior.add_bos);
                }
                if behavior.has_add_eos {
                    default_add_eos = Some(behavior.add_eos);
                }
            }
        }

        Ok(Self {
            ptr: tokenizer_ptr,
            default_add_bos,
            default_add_eos,
        })
    }

    pub fn vocab_size(&self) -> usize {
        unsafe { marmot_tokenizer_vocab_size(self.ptr) }
    }

    pub fn special_tokens(&self) -> Result<SpecialTokens> {
        let mut special = std::mem::MaybeUninit::<marmot_tokenizer_special_ids_t>::uninit();
        unsafe {
            check_error(marmot_tokenizer_get_special_ids(
                self.ptr,
                special.as_mut_ptr(),
            ))?;
            let special = special.assume_init();
            Ok(SpecialTokens {
                bos_id: if special.has_bos {
                    Some(special.bos_id)
                } else {
                    None
                },
                eos_id: if special.has_eos {
                    Some(special.eos_id)
                } else {
                    None
                },
                unk_id: if special.has_unk {
                    Some(special.unk_id)
                } else {
                    None
                },
                pad_id: if special.has_pad {
                    Some(special.pad_id)
                } else {
                    None
                },
            })
        }
    }

    pub fn piece_to_token(&self, piece: &str) -> Result<Option<i32>> {
        let c_piece = CString::new(piece.as_bytes()).map_err(|_| MarmotError {
            code: MARMOT_ERROR_INVALID_ARGUMENT,
            message: "Invalid piece".to_string(),
        })?;
        let mut token_id: marmot_token_id_t = MARMOT_TOKEN_ID_INVALID;
        let status = unsafe {
            marmot_tokenizer_piece_to_token(
                self.ptr,
                c_piece.as_ptr(),
                piece.len(),
                &mut token_id,
            )
        };
        if status == MARMOT_OK {
            return Ok(Some(token_id));
        }
        if status == MARMOT_ERROR_INVALID_ARGUMENT {
            return Ok(None);
        }
        check_error(status)?;
        Ok(None)
    }

    /// Convert a token ID to its string representation (piece).
    pub fn token_to_piece(&self, token_id: i32) -> Result<Option<String>> {
        // First call to get the required length
        let mut len: usize = 0;
        let status = unsafe {
            marmot_tokenizer_token_to_piece(self.ptr, token_id, ptr::null_mut(), &mut len)
        };
        check_error(status)?;

        if len == 0 {
            return Ok(Some(String::new()));
        }

        // Allocate buffer and get the piece
        let mut buffer = vec![0u8; len + 1];
        unsafe {
            check_error(marmot_tokenizer_token_to_piece(
                self.ptr,
                token_id,
                buffer.as_mut_ptr() as *mut i8,
                &mut len,
            ))?;
        }
        truncate_c_string(&mut buffer, len);
        Ok(Some(String::from_utf8_lossy(&buffer).into_owned()))
    }

    /// Get the BOS token as a string, if available.
    pub fn bos_token_string(&self) -> Option<String> {
        let special = self.special_tokens().ok()?;
        let bos_id = special.bos_id?;
        self.token_to_piece(bos_id).ok().flatten()
    }

    /// Get the EOS token as a string, if available.
    pub fn eos_token_string(&self) -> Option<String> {
        let special = self.special_tokens().ok()?;
        let eos_id = special.eos_id?;
        self.token_to_piece(eos_id).ok().flatten()
    }

    pub fn encode(&self, text: &str, add_bos: bool, add_eos: bool) -> Result<Vec<i32>> {
        let mut opts = std::mem::MaybeUninit::<marmot_tokenizer_encode_options_t>::uninit();
        let add_bos = self.default_add_bos.unwrap_or(add_bos);
        let add_eos = self.default_add_eos.unwrap_or(add_eos);
        unsafe {
            check_error(marmot_tokenizer_encode_options_init(opts.as_mut_ptr()))?;
            let opts_ptr = opts.as_mut_ptr();
            (*opts_ptr).add_bos = add_bos;
            (*opts_ptr).add_eos = add_eos;
            (*opts_ptr).allow_special = true;
        }

        // First call to get the required length
        let mut len: usize = 0;
        unsafe {
            let err = marmot_tokenizer_encode(
                self.ptr,
                text.as_ptr() as *const i8,
                text.len(),
                opts.as_ptr(),
                ptr::null_mut(),
                &mut len,
            );
            // Expect buffer too small error, which tells us the required size
            if err != MARMOT_OK && err != MARMOT_ERROR_INVALID_ARGUMENT {
                check_error(err)?;
            }
        }

        if len == 0 {
            return Ok(Vec::new());
        }

        // Allocate buffer and encode
        let mut tokens = vec![0i32; len];
        unsafe {
            check_error(marmot_tokenizer_encode(
                self.ptr,
                text.as_ptr() as *const i8,
                text.len(),
                opts.as_ptr(),
                tokens.as_mut_ptr(),
                &mut len,
            ))?;
        }
        tokens.truncate(len);
        Ok(tokens)
    }

    pub fn chat_template(&self) -> Result<Option<String>> {
        // First call to get length
        let mut len: usize = 0;
        unsafe {
            let err = marmot_tokenizer_chat_template(self.ptr, ptr::null_mut(), &mut len);
            if err == MARMOT_ERROR_NOT_IMPLEMENTED {
                // No chat template in this model
                return Ok(None);
            }
            if err != MARMOT_OK && err != MARMOT_ERROR_INVALID_ARGUMENT {
                check_error(err)?;
            }
        }

        if len == 0 {
            return Ok(None);
        }

        // Allocate and get template
        let mut buffer = vec![0u8; len + 1];
        unsafe {
            check_error(marmot_tokenizer_chat_template(
                self.ptr,
                buffer.as_mut_ptr() as *mut i8,
                &mut len,
            ))?;
        }
        truncate_c_string(&mut buffer, len);
        let template = String::from_utf8_lossy(&buffer).into_owned();
        if template.trim().is_empty() {
            return Ok(None);
        }
        Ok(Some(template))
    }

    pub fn decode(&self, tokens: &[i32]) -> Result<String> {
        self.decode_with_options(tokens, false, true)
    }

    pub fn decode_streaming(&self, tokens: &[i32]) -> Result<String> {
        self.decode_with_options(tokens, true, false)
    }

    fn decode_with_options(&self, tokens: &[i32], skip_special: bool, strip_space_prefix: bool) -> Result<String> {
        if tokens.is_empty() {
            return Ok(String::new());
        }

        let mut opts = std::mem::MaybeUninit::<marmot_tokenizer_decode_options_t>::uninit();
        unsafe {
            check_error(marmot_tokenizer_decode_options_init(opts.as_mut_ptr()))?;
            (*opts.as_mut_ptr()).skip_special = skip_special;
            (*opts.as_mut_ptr()).strip_space_prefix = strip_space_prefix;
        }

        // First call to get the required length
        let mut len: usize = 0;
        unsafe {
            let err = marmot_tokenizer_decode(
                self.ptr,
                tokens.as_ptr(),
                tokens.len(),
                opts.as_ptr(),
                ptr::null_mut(),
                &mut len,
            );
            if err != MARMOT_OK && err != MARMOT_ERROR_INVALID_ARGUMENT {
                check_error(err)?;
            }
        }

        if len == 0 {
            return Ok(String::new());
        }

        // Allocate buffer and decode
        let mut buffer = vec![0u8; len + 1];
        unsafe {
            check_error(marmot_tokenizer_decode(
                self.ptr,
                tokens.as_ptr(),
                tokens.len(),
                opts.as_ptr(),
                buffer.as_mut_ptr() as *mut i8,
                &mut len,
            ))?;
        }
        truncate_c_string(&mut buffer, len);
        Ok(String::from_utf8_lossy(&buffer).into_owned())
    }
}

impl Drop for Tokenizer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { marmot_tokenizer_destroy(self.ptr) };
        }
    }
}

unsafe impl Send for Tokenizer {}
unsafe impl Sync for Tokenizer {}

/// Named sampling profiles for common use cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingPreset {
    /// Greedy decoding for debugging correctness issues.
    Debug,
    /// Low-slop profile for factual Q&A, coding, and summarization.
    Assistant,
    /// Exploratory profile for creative and open-ended generation.
    Creative,
}

impl SamplingPreset {
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "debug" => Some(Self::Debug),
            "assistant" => Some(Self::Assistant),
            "creative" => Some(Self::Creative),
            _ => None,
        }
    }
}

/// Sampling options for text generation
#[derive(Debug, Clone)]
pub struct SamplingOptions {
    pub seed: u64,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub suppress_special_tokens: bool,
}

impl SamplingOptions {
    pub fn from_preset(preset: SamplingPreset) -> Self {
        match preset {
            SamplingPreset::Debug => Self {
                seed: 42,
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                repetition_penalty: 1.0,
                suppress_special_tokens: true,
            },
            SamplingPreset::Assistant => Self {
                seed: 42,
                temperature: 0.3,
                top_k: 30,
                top_p: 0.9,
                min_p: 0.0,
                repetition_penalty: 1.05,
                suppress_special_tokens: true,
            },
            SamplingPreset::Creative => Self {
                seed: 42,
                temperature: 0.8,
                top_k: 40,
                top_p: 0.95,
                min_p: 0.05,
                repetition_penalty: 1.1,
                suppress_special_tokens: true,
            },
        }
    }
}

impl Default for SamplingOptions {
    fn default() -> Self {
        Self::from_preset(SamplingPreset::Assistant)
    }
}

/// Generation options
#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub max_new_tokens: usize,
    pub stop_on_eos: bool,
    pub stop_tokens: Vec<i32>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            stop_on_eos: true,
            stop_tokens: Vec::new(),
        }
    }
}

/// Request state for generation requests
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestState {
    Invalid,
    Pending,
    Prefill,
    Decoding,
    Done,
    Failed,
    Canceled,
}

impl From<marmot_llm_request_state_t> for RequestState {
    fn from(state: marmot_llm_request_state_t) -> Self {
        match state {
            marmot_llm_request_state_t::MARMOT_LLM_REQUEST_STATE_INVALID => RequestState::Invalid,
            marmot_llm_request_state_t::MARMOT_LLM_REQUEST_STATE_PENDING => RequestState::Pending,
            marmot_llm_request_state_t::MARMOT_LLM_REQUEST_STATE_PREFILL => RequestState::Prefill,
            marmot_llm_request_state_t::MARMOT_LLM_REQUEST_STATE_DECODING => RequestState::Decoding,
            marmot_llm_request_state_t::MARMOT_LLM_REQUEST_STATE_DONE => RequestState::Done,
            marmot_llm_request_state_t::MARMOT_LLM_REQUEST_STATE_FAILED => RequestState::Failed,
            marmot_llm_request_state_t::MARMOT_LLM_REQUEST_STATE_CANCELED => RequestState::Canceled,
        }
    }
}

struct TokenCollector {
    tokens: Mutex<Vec<i32>>,
}

impl TokenCollector {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            tokens: Mutex::new(Vec::new()),
        })
    }
}

unsafe extern "C" fn serving_on_token(user_data: *mut c_void, token_id: marmot_token_id_t) {
    if user_data.is_null() {
        return;
    }
    let collector = unsafe { &*(user_data as *const TokenCollector) };
    if let Ok(mut tokens) = collector.tokens.lock() {
        tokens.push(token_id);
    }
}

struct CollectorEntry {
    collector: Arc<TokenCollector>,
    raw_ptr: *const TokenCollector,
}

/// Serving Engine for paged attention (persistent, multi-request)
pub struct ServingEngine {
    ptr: *mut marmot_serving_engine_t,
    collectors: HashMap<u64, CollectorEntry>,
}

// SAFETY: The marmot C library is designed to be thread-safe when used correctly.
// ServingEngine instances are only accessed through &mut self, ensuring exclusive access.
unsafe impl Send for ServingEngine {}
unsafe impl Sync for ServingEngine {}

impl ServingEngine {
    pub fn new(ctx: &Context, model: &Model) -> Result<Self> {
        let mut opts = std::mem::MaybeUninit::<marmot_serving_engine_options_t>::uninit();
        unsafe {
            check_error(marmot_serving_engine_options_init(opts.as_mut_ptr()))?;
        }

        let info = model.info()?;
        unsafe {
            let opts_ptr = opts.as_mut_ptr();
            if info.context_length > 0 {
                if (*opts_ptr).max_seq_len > info.context_length {
                    (*opts_ptr).max_seq_len = info.context_length;
                }
                if (*opts_ptr).max_num_tokens > (*opts_ptr).max_seq_len {
                    (*opts_ptr).max_num_tokens = (*opts_ptr).max_seq_len;
                }
                if (*opts_ptr).max_batch_seqs > (*opts_ptr).max_num_tokens {
                    (*opts_ptr).max_batch_seqs = (*opts_ptr).max_num_tokens;
                }
            }
        }
        let arch_name = CString::new(info.architecture.clone()).map_err(|_| MarmotError {
            code: MARMOT_ERROR_INVALID_ARGUMENT,
            message: "Invalid architecture name".to_string(),
        })?;
        let arch_id = unsafe { marmot_architecture_from_string(arch_name.as_ptr()) };
        let kv_dtype = unsafe { marmot_activation_dtype_for_architecture(arch_id, ctx.backend_type()) };
        unsafe {
            (*opts.as_mut_ptr()).kv_dtype = kv_dtype;
        }

        let mut engine_ptr: *mut marmot_serving_engine_t = ptr::null_mut();
        unsafe {
            check_error(marmot_serving_engine_create(
                ctx.as_ptr(),
                model.as_ptr(),
                opts.as_ptr(),
                &mut engine_ptr,
            ))?;
        }

        Ok(Self {
            ptr: engine_ptr,
            collectors: HashMap::new(),
        })
    }

    /// Submit a generation request
    pub fn submit(
        &mut self,
        prompt_tokens: &[i32],
        gen_opts: &GenerateOptions,
        sampling: &SamplingOptions,
    ) -> Result<u64> {
        let mut raw_gen_opts = std::mem::MaybeUninit::<marmot_llm_generate_options_t>::uninit();
        let mut raw_sampling = std::mem::MaybeUninit::<marmot_llm_sampling_options_t>::uninit();

        let collector = TokenCollector::new();
        let raw_ptr = Arc::into_raw(Arc::clone(&collector));

        unsafe {
            check_error(marmot_llm_generate_options_init(raw_gen_opts.as_mut_ptr()))?;
            check_error(marmot_llm_sampling_options_init(raw_sampling.as_mut_ptr()))?;

            let gen_ptr = raw_gen_opts.as_mut_ptr();
            (*gen_ptr).max_new_tokens = gen_opts.max_new_tokens;
            (*gen_ptr).stop_on_eos = gen_opts.stop_on_eos;
            if !gen_opts.stop_tokens.is_empty() {
                (*gen_ptr).stop_tokens = gen_opts.stop_tokens.as_ptr();
                (*gen_ptr).stop_tokens_len = gen_opts.stop_tokens.len();
            }
            (*gen_ptr).on_token = Some(serving_on_token);
            (*gen_ptr).user_data = raw_ptr as *mut c_void;

            let samp_ptr = raw_sampling.as_mut_ptr();
            (*samp_ptr).seed = sampling.seed;
            (*samp_ptr).temperature = sampling.temperature;
            (*samp_ptr).top_k = sampling.top_k;
            (*samp_ptr).top_p = sampling.top_p;
            (*samp_ptr).min_p = sampling.min_p;
            (*samp_ptr).repetition_penalty = sampling.repetition_penalty;
            if sampling.suppress_special_tokens {
                (*samp_ptr).flags |= MARMOT_LLM_SAMPLING_FLAG_SUPPRESS_SPECIAL_TOKENS;
            }
        }

        let mut request_id: u64 = 0;
        let submit_status = unsafe {
            marmot_serving_engine_submit(
                self.ptr,
                prompt_tokens.as_ptr(),
                prompt_tokens.len(),
                raw_gen_opts.as_ptr(),
                raw_sampling.as_ptr(),
                &mut request_id,
            )
        };
        if submit_status != MARMOT_OK {
            unsafe {
                let _ = Arc::from_raw(raw_ptr);
            }
            return Err(check_error(submit_status).unwrap_err());
        }

        self.collectors.insert(
            request_id,
            CollectorEntry {
                collector,
                raw_ptr,
            },
        );

        Ok(request_id)
    }

    /// Step the serving engine
    pub fn step(&mut self, max_steps: usize) -> Result<usize> {
        let mut steps_done: usize = 0;
        unsafe {
            check_error(marmot_serving_engine_step(self.ptr, max_steps, &mut steps_done))?;
        }
        Ok(steps_done)
    }

    /// Get the state of a request
    pub fn request_state(&self, request_id: u64) -> RequestState {
        let state = unsafe { marmot_serving_engine_request_state(self.ptr, request_id) };
        RequestState::from(state)
    }

    /// Get tokens generated for a request
    pub fn get_tokens(&self, request_id: u64) -> Result<Vec<i32>> {
        let entry = match self.collectors.get(&request_id) {
            Some(entry) => entry,
            None => return Ok(Vec::new()),
        };
        let tokens = entry.collector.tokens.lock().map_err(|_| MarmotError {
            code: MARMOT_ERROR_INVALID_OPERATION,
            message: "Token collector lock poisoned".to_string(),
        })?;
        Ok(tokens.clone())
    }

    /// Cancel a request
    pub fn cancel(&mut self, request_id: u64) -> Result<()> {
        unsafe {
            check_error(marmot_serving_engine_request_cancel(self.ptr, request_id))?;
        }
        Ok(())
    }

    /// Release a completed request
    pub fn release(&mut self, request_id: u64) -> Result<()> {
        unsafe {
            check_error(marmot_serving_engine_request_release(self.ptr, request_id))?;
        }
        if let Some(entry) = self.collectors.remove(&request_id) {
            unsafe {
                let _ = Arc::from_raw(entry.raw_ptr);
            }
        }
        Ok(())
    }
}

impl Drop for ServingEngine {
    fn drop(&mut self) {
        for (_, entry) in self.collectors.drain() {
            unsafe {
                let _ = Arc::from_raw(entry.raw_ptr);
            }
        }
        if !self.ptr.is_null() {
            unsafe { marmot_serving_engine_destroy(self.ptr) };
        }
    }
}

