//! Raw FFI bindings to libmarmot
//!
//! These are low-level unsafe bindings. Use the safe wrappers in `super::safe_wrappers` instead.
//! Not all types/constants are consumed by Rust callers — they mirror the full C API surface.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::ffi::{c_char, c_void};

// Error codes
pub type marmot_error_t = i32;

pub const MARMOT_OK: marmot_error_t = 0;
pub const MARMOT_ERROR_INVALID_ARGUMENT: marmot_error_t = -1;
pub const MARMOT_ERROR_OUT_OF_MEMORY: marmot_error_t = -2;
pub const MARMOT_ERROR_DEVICE_NOT_AVAILABLE: marmot_error_t = -3;
pub const MARMOT_ERROR_BACKEND_INIT_FAILED: marmot_error_t = -4;
pub const MARMOT_ERROR_INVALID_OPERATION: marmot_error_t = -5;
pub const MARMOT_ERROR_UNSUPPORTED_DTYPE: marmot_error_t = -6;
pub const MARMOT_ERROR_DIMENSION_MISMATCH: marmot_error_t = -7;
pub const MARMOT_ERROR_NOT_IMPLEMENTED: marmot_error_t = -8;

// Backend types
pub type marmot_backend_type_t = i32;

pub const MARMOT_BACKEND_CPU: marmot_backend_type_t = 0;
pub const MARMOT_BACKEND_METAL: marmot_backend_type_t = 1;

// Architecture types
pub type marmot_architecture_t = i32;

// Dtype types
pub type marmot_dtype_t = i32;
pub const MARMOT_DTYPE_FLOAT32: marmot_dtype_t = 0;
pub const MARMOT_DTYPE_FLOAT16: marmot_dtype_t = 1;
pub const MARMOT_DTYPE_BFLOAT16: marmot_dtype_t = 2;

// Rope types
pub type marmot_rope_type_t = i32;

// Rope scaling types
pub type marmot_rope_scaling_type_t = i32;

// Token type
pub type marmot_token_id_t = i32;
pub const MARMOT_TOKEN_ID_INVALID: marmot_token_id_t = -1;

// Opaque types
#[repr(C)]
pub struct marmot_context_t {
    _private: [u8; 0],
}

#[repr(C)]
pub struct marmot_model_t {
    _private: [u8; 0],
}

#[repr(C)]
pub struct marmot_tokenizer_t {
    _private: [u8; 0],
}

#[repr(C)]
pub struct marmot_serving_engine_t {
    _private: [u8; 0],
}

// Model options
#[repr(C)]
pub struct marmot_model_options_t {
    pub struct_size: u32,
    pub struct_version: u32,
    pub flags: u64,
    pub pnext: *const c_void,
    pub reserved: [u64; 4],
}

// Model info
#[repr(C)]
pub struct marmot_model_info_t {
    pub architecture: [c_char; 32],
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

// Tokenizer options
#[repr(C)]
pub struct marmot_tokenizer_options_t {
    pub struct_size: u32,
    pub struct_version: u32,
    pub flags: u64,
    pub pnext: *const c_void,
    pub reserved: [u64; 4],
}

#[repr(C)]
pub struct marmot_tokenizer_encode_options_t {
    pub struct_size: u32,
    pub struct_version: u32,
    pub add_bos: bool,
    pub add_eos: bool,
    pub allow_special: bool,
    pub max_tokens: usize,
    pub pnext: *const c_void,
    pub reserved: [u64; 4],
}

#[repr(C)]
pub struct marmot_tokenizer_decode_options_t {
    pub struct_size: u32,
    pub struct_version: u32,
    pub skip_special: bool,
    pub strip_space_prefix: bool,
    pub pnext: *const c_void,
    pub reserved: [u64; 4],
}

#[repr(C)]
pub struct marmot_tokenizer_special_ids_t {
    pub has_bos: bool,
    pub has_eos: bool,
    pub has_unk: bool,
    pub has_pad: bool,
    pub bos_id: marmot_token_id_t,
    pub eos_id: marmot_token_id_t,
    pub unk_id: marmot_token_id_t,
    pub pad_id: marmot_token_id_t,
}

#[repr(C)]
pub struct marmot_tokenizer_behavior_t {
    pub has_add_bos: bool,
    pub add_bos: bool,
    pub has_add_eos: bool,
    pub add_eos: bool,
}

pub const MARMOT_LLM_SAMPLING_FLAG_SUPPRESS_SPECIAL_TOKENS: u64 = 1 << 0;

// Callback types
pub type marmot_llm_progress_callback_t = Option<unsafe extern "C" fn(*mut c_void, usize, usize)>;
pub type marmot_llm_token_callback_t = Option<unsafe extern "C" fn(*mut c_void, marmot_token_id_t)>;

#[repr(C)]
pub struct marmot_llm_generate_options_t {
    pub struct_size: u32,
    pub struct_version: u32,
    pub flags: u64,
    pub max_new_tokens: usize,
    pub stop_on_eos: bool,
    pub stop_tokens: *const marmot_token_id_t,
    pub stop_tokens_len: usize,
    pub prefill_progress: marmot_llm_progress_callback_t,
    pub on_token: marmot_llm_token_callback_t,
    pub user_data: *mut c_void,
    pub pnext: *const c_void,
    pub reserved: [u64; 4],
}

#[repr(C)]
pub struct marmot_llm_sampling_options_t {
    pub struct_size: u32,
    pub struct_version: u32,
    pub flags: u64,
    pub seed: u64,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub pnext: *const c_void,
    pub reserved: [u64; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum marmot_llm_request_state_t {
    MARMOT_LLM_REQUEST_STATE_INVALID = 0,
    MARMOT_LLM_REQUEST_STATE_PENDING = 1,
    MARMOT_LLM_REQUEST_STATE_PREFILL = 2,
    MARMOT_LLM_REQUEST_STATE_DECODING = 3,
    MARMOT_LLM_REQUEST_STATE_DONE = 4,
    MARMOT_LLM_REQUEST_STATE_FAILED = 5,
    MARMOT_LLM_REQUEST_STATE_CANCELED = 6,
}

// Serving Engine types
pub type marmot_request_id_t = u64;

#[repr(C)]
pub struct marmot_serving_engine_batch_view_t {
    pub token_count: usize,
    pub sample_count: usize,
    pub token_ids: *const marmot_token_id_t,
    pub token_meta: *const u32,
    pub sample_indices: *const u32,
    pub sample_request_ids: *const marmot_request_id_t,
}

#[repr(C)]
pub struct marmot_serving_engine_options_t {
    pub struct_size: u32,
    pub struct_version: u32,
    pub flags: u64,
    pub max_seqs: usize,
    pub max_batch_seqs: usize,
    pub max_num_tokens: usize,
    pub max_seq_len: usize,
    pub block_size: usize,
    pub num_kv_blocks: usize,
    pub num_swap_blocks: usize,
    pub kv_dtype: marmot_dtype_t,
    pub kv_block_watermark: f32,
    pub prefill_chunk_size: usize,
    pub pnext: *const c_void,
    pub reserved: [u64; 4],
}

#[repr(C)]
pub struct marmot_serving_request_ext_t {
    pub struct_size: u32,
    pub struct_version: u32,
    pub flags: u64,
    pub priority: i32,
    pub cache_salt: *const c_char,
    pub cache_salt_len: usize,
    pub retention_blocks: usize,
    pub num_samples: usize,
    pub sample_user_data: *const *mut c_void,
    pub sample_user_data_len: usize,
    pub out_request_ids: *mut marmot_request_id_t,
    pub out_request_ids_capacity: usize,
    pub pnext: *const c_void,
    pub reserved: [u64; 4],
}

// External functions
#[link(name = "marmot")]
extern "C" {
    // Context management
    pub fn marmot_init(backend: marmot_backend_type_t) -> *mut marmot_context_t;
    pub fn marmot_destroy(ctx: *mut marmot_context_t);
    pub fn marmot_context_get_backend(ctx: *const marmot_context_t) -> marmot_backend_type_t;

    // Error handling
    pub fn marmot_error_string(error: marmot_error_t) -> *const c_char;
    pub fn marmot_get_last_error() -> marmot_error_t;
    pub fn marmot_get_last_error_detail() -> *const c_char;
    // Model loading
    pub fn marmot_model_options_init(opts: *mut marmot_model_options_t) -> marmot_error_t;
    pub fn marmot_model_load_file(
        path: *const c_char,
        opts: *const marmot_model_options_t,
        out_model: *mut *mut marmot_model_t,
    ) -> marmot_error_t;
    pub fn marmot_model_destroy(model: *mut marmot_model_t);
    pub fn marmot_model_get_info(
        model: *const marmot_model_t,
        out_info: *mut marmot_model_info_t,
    ) -> marmot_error_t;
    // Architecture helpers
    pub fn marmot_architecture_from_string(name: *const c_char) -> marmot_architecture_t;
    pub fn marmot_activation_dtype_for_architecture(
        arch: marmot_architecture_t,
        backend: marmot_backend_type_t,
    ) -> marmot_dtype_t;
    // Tokenizer
    pub fn marmot_tokenizer_options_init(opts: *mut marmot_tokenizer_options_t) -> marmot_error_t;
    pub fn marmot_tokenizer_encode_options_init(
        opts: *mut marmot_tokenizer_encode_options_t,
    ) -> marmot_error_t;
    pub fn marmot_tokenizer_decode_options_init(
        opts: *mut marmot_tokenizer_decode_options_t,
    ) -> marmot_error_t;
    pub fn marmot_tokenizer_create_from_gguf_file(
        path: *const c_char,
        opts: *const marmot_tokenizer_options_t,
        out_tokenizer: *mut *mut marmot_tokenizer_t,
    ) -> marmot_error_t;
    pub fn marmot_tokenizer_destroy(tokenizer: *mut marmot_tokenizer_t);
    pub fn marmot_tokenizer_vocab_size(tokenizer: *const marmot_tokenizer_t) -> usize;
    pub fn marmot_tokenizer_get_special_ids(
        tokenizer: *const marmot_tokenizer_t,
        out_special_ids: *mut marmot_tokenizer_special_ids_t,
    ) -> marmot_error_t;
    pub fn marmot_tokenizer_get_behavior(
        tokenizer: *const marmot_tokenizer_t,
        out_behavior: *mut marmot_tokenizer_behavior_t,
    ) -> marmot_error_t;
    pub fn marmot_tokenizer_piece_to_token(
        tokenizer: *const marmot_tokenizer_t,
        piece: *const c_char,
        piece_len: usize,
        out_token_id: *mut marmot_token_id_t,
    ) -> marmot_error_t;
    pub fn marmot_tokenizer_token_to_piece(
        tokenizer: *const marmot_tokenizer_t,
        token_id: marmot_token_id_t,
        out_piece: *mut c_char,
        inout_len: *mut usize,
    ) -> marmot_error_t;
    pub fn marmot_tokenizer_encode(
        tokenizer: *const marmot_tokenizer_t,
        text: *const c_char,
        text_len: usize,
        opts: *const marmot_tokenizer_encode_options_t,
        out_token_ids: *mut marmot_token_id_t,
        inout_len: *mut usize,
    ) -> marmot_error_t;
    pub fn marmot_tokenizer_decode(
        tokenizer: *const marmot_tokenizer_t,
        token_ids: *const marmot_token_id_t,
        token_ids_len: usize,
        opts: *const marmot_tokenizer_decode_options_t,
        out_text: *mut c_char,
        inout_len: *mut usize,
    ) -> marmot_error_t;
    pub fn marmot_tokenizer_chat_template(
        tokenizer: *const marmot_tokenizer_t,
        out_template: *mut c_char,
        inout_len: *mut usize,
    ) -> marmot_error_t;
    pub fn marmot_llm_generate_options_init(
        opts: *mut marmot_llm_generate_options_t,
    ) -> marmot_error_t;
    pub fn marmot_llm_sampling_options_init(
        opts: *mut marmot_llm_sampling_options_t,
    ) -> marmot_error_t;

    // Serving Engine (paged attention API)
    pub fn marmot_serving_engine_options_init(
        opts: *mut marmot_serving_engine_options_t,
    ) -> marmot_error_t;
    pub fn marmot_serving_engine_create(
        ctx: *const marmot_context_t,
        model: *const marmot_model_t,
        opts: *const marmot_serving_engine_options_t,
        out_engine: *mut *mut marmot_serving_engine_t,
    ) -> marmot_error_t;
    pub fn marmot_serving_engine_destroy(engine: *mut marmot_serving_engine_t);
    pub fn marmot_serving_engine_submit(
        engine: *mut marmot_serving_engine_t,
        prompt_tokens: *const marmot_token_id_t,
        prompt_len: usize,
        gen_opts: *const marmot_llm_generate_options_t,
        sampling_opts: *const marmot_llm_sampling_options_t,
        out_request_id: *mut marmot_request_id_t,
    ) -> marmot_error_t;
    pub fn marmot_serving_engine_step(
        engine: *mut marmot_serving_engine_t,
        max_steps: usize,
        out_steps_done: *mut usize,
    ) -> marmot_error_t;
    pub fn marmot_serving_engine_request_state(
        engine: *const marmot_serving_engine_t,
        request_id: marmot_request_id_t,
    ) -> marmot_llm_request_state_t;
    pub fn marmot_serving_engine_request_cancel(
        engine: *mut marmot_serving_engine_t,
        request_id: marmot_request_id_t,
    ) -> marmot_error_t;
    pub fn marmot_serving_engine_request_release(
        engine: *mut marmot_serving_engine_t,
        request_id: marmot_request_id_t,
    ) -> marmot_error_t;
    pub fn marmot_serving_engine_last_batch(
        engine: *const marmot_serving_engine_t,
        out_batch: *mut marmot_serving_engine_batch_view_t,
    ) -> marmot_error_t;
}
