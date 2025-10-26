//! Custom Jinja2 filters for HuggingFace template compatibility.

use minijinja::{Environment, Error, ErrorKind, Value, value::{ValueKind, from_args}};

/// Register custom filters needed for HuggingFace chat templates.
pub fn register_filters(env: &mut Environment<'static>) {
    env.add_filter("tojson", filter_tojson);
    env.add_function("raise_exception", fn_raise_exception);

    // Add string method support for HuggingFace templates (Qwen3, etc.)
    // These templates use Python string methods like .split(), .strip(), etc.
    env.set_unknown_method_callback(string_method_callback);
}

/// Handle string methods that HuggingFace templates expect.
/// Python Jinja2 exposes string methods directly; minijinja doesn't by default.
fn string_method_callback(
    _state: &minijinja::State,
    value: &Value,
    method: &str,
    args: &[Value],
) -> Result<Value, Error> {
    // Only handle string values
    if value.kind() != ValueKind::String {
        return Err(Error::new(
            ErrorKind::UnknownMethod,
            format!("{} has no method named {}", value.kind(), method),
        ));
    }

    let s = value.as_str().unwrap_or("");

    match method {
        "split" => {
            let (sep,): (&str,) = from_args(args)?;
            let parts: Vec<Value> = s.split(sep).map(Value::from).collect();
            Ok(Value::from(parts))
        }
        "strip" => {
            let chars: Option<&str> = if args.is_empty() {
                None
            } else {
                let (c,): (&str,) = from_args(args)?;
                Some(c)
            };
            let result = match chars {
                Some(c) => s.trim_matches(|ch: char| c.contains(ch)),
                None => s.trim(),
            };
            Ok(Value::from(result))
        }
        "lstrip" => {
            let chars: Option<&str> = if args.is_empty() {
                None
            } else {
                let (c,): (&str,) = from_args(args)?;
                Some(c)
            };
            let result = match chars {
                Some(c) => s.trim_start_matches(|ch: char| c.contains(ch)),
                None => s.trim_start(),
            };
            Ok(Value::from(result))
        }
        "rstrip" => {
            let chars: Option<&str> = if args.is_empty() {
                None
            } else {
                let (c,): (&str,) = from_args(args)?;
                Some(c)
            };
            let result = match chars {
                Some(c) => s.trim_end_matches(|ch: char| c.contains(ch)),
                None => s.trim_end(),
            };
            Ok(Value::from(result))
        }
        "startswith" => {
            let (prefix,): (&str,) = from_args(args)?;
            Ok(Value::from(s.starts_with(prefix)))
        }
        "endswith" => {
            let (suffix,): (&str,) = from_args(args)?;
            Ok(Value::from(s.ends_with(suffix)))
        }
        "upper" => {
            let _: () = from_args(args)?;
            Ok(Value::from(s.to_uppercase()))
        }
        "lower" => {
            let _: () = from_args(args)?;
            Ok(Value::from(s.to_lowercase()))
        }
        _ => Err(Error::new(
            ErrorKind::UnknownMethod,
            format!("string has no method named {}", method),
        )),
    }
}

/// Convert a value to JSON string.
fn filter_tojson(value: Value) -> Result<String, Error> {
    serde_json::to_string(&value)
        .map_err(|e| Error::new(ErrorKind::InvalidOperation, format!("JSON serialization failed: {}", e)))
}

/// Raise a template exception with a custom message.
/// Used by templates to enforce constraints (e.g., alternating roles).
fn fn_raise_exception(msg: String) -> Result<String, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, msg))
}

#[cfg(test)]
mod tests {
    use super::*;
    use minijinja::context;

    #[test]
    fn test_tojson() {
        let mut env = Environment::new();
        register_filters(&mut env);

        env.add_template("test", "{{ value | tojson }}").unwrap();
        let tmpl = env.get_template("test").unwrap();

        let result = tmpl.render(context!(value => "hello")).unwrap();
        assert_eq!(result, r#""hello""#);

        let result = tmpl.render(context!(value => vec!["a", "b"])).unwrap();
        assert_eq!(result, r#"["a","b"]"#);
    }

    #[test]
    fn test_raise_exception() {
        let mut env = Environment::new();
        register_filters(&mut env);

        env.add_template("test", "{{ raise_exception('test error') }}")
            .unwrap();
        let tmpl = env.get_template("test").unwrap();

        let result = tmpl.render(context!());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("test error"));
    }

    #[test]
    fn test_string_split() {
        let mut env = Environment::new();
        register_filters(&mut env);

        env.add_template("test", "{{ value.split('</think>')[-1] }}")
            .unwrap();
        let tmpl = env.get_template("test").unwrap();

        let result = tmpl.render(context!(value => "before</think>after")).unwrap();
        assert_eq!(result, "after");
    }

    #[test]
    fn test_string_strip() {
        let mut env = Environment::new();
        register_filters(&mut env);

        env.add_template("test", "[{{ value.strip() }}]").unwrap();
        let tmpl = env.get_template("test").unwrap();

        let result = tmpl.render(context!(value => "  hello  ")).unwrap();
        assert_eq!(result, "[hello]");
    }

    #[test]
    fn test_string_strip_chars() {
        let mut env = Environment::new();
        register_filters(&mut env);

        env.add_template("test", "[{{ value.strip('\n') }}]").unwrap();
        let tmpl = env.get_template("test").unwrap();

        let result = tmpl.render(context!(value => "\n\nhello\n")).unwrap();
        assert_eq!(result, "[hello]");
    }

    #[test]
    fn test_string_lstrip() {
        let mut env = Environment::new();
        register_filters(&mut env);

        env.add_template("test", "[{{ value.lstrip() }}]").unwrap();
        let tmpl = env.get_template("test").unwrap();

        let result = tmpl.render(context!(value => "\n  hello")).unwrap();
        assert_eq!(result, "[hello]");
    }

    #[test]
    fn test_string_rstrip() {
        let mut env = Environment::new();
        register_filters(&mut env);

        env.add_template("test", "[{{ value.rstrip() }}]").unwrap();
        let tmpl = env.get_template("test").unwrap();

        let result = tmpl.render(context!(value => "hello  \n")).unwrap();
        assert_eq!(result, "[hello]");
    }

    #[test]
    fn test_qwen3_split_pattern() {
        // This is the exact pattern from Qwen3 chat template
        let mut env = Environment::new();
        register_filters(&mut env);

        let template = r#"
{%- set content = message.content.split('</think>')[-1].lstrip('\n') -%}
{{ content }}
"#;
        env.add_template("test", template).unwrap();
        let tmpl = env.get_template("test").unwrap();

        use std::collections::HashMap;
        let mut message = HashMap::new();
        message.insert("content", "<think>reasoning</think>\n\nactual response");

        let result = tmpl.render(context!(message => message)).unwrap();
        assert_eq!(result.trim(), "actual response");
    }
}
