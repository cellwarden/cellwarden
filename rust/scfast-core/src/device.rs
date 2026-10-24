//! Compute device selection (docs/scfast-integration.md Phase 6).
//!
//! `Device::Cpu` is always available and is the reference implementation. `Device::Cuda(i)`
//! exists only when the crate was built with the `cuda` feature **and** the CUDA driver,
//! and NVRTC shared libraries can be loaded at run time **and** device `i` is
//! visible. Without the feature, every CUDA query answers "not built" without touching any
//! library, so the default build is identical to the pre-Phase-6 CPU engine.

use crate::error::{Error, Result};

/// Where a kernel runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Device {
    /// rayon + faer, any machine.
    Cpu,
    /// CUDA device with this ordinal (feature `cuda`).
    Cuda(usize),
}

impl Device {
    /// CUDA device 0 when the `cuda` feature is compiled in, the driver loads and at least
    /// one device is visible; CPU otherwise. Never fails and never panics.
    pub fn auto() -> Device {
        if cuda_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        }
    }

    /// Parse a user-facing device string: `"auto"`, `"cpu"`, `"cuda"` / `"gpu"` (device 0)
    /// or `"cuda:N"`. Case-insensitive.
    ///
    /// `"cuda"` with the feature not compiled in, or with no usable device, is
    /// [`Error::Gpu`] with a message that says which; an unknown string is
    /// [`Error::InvalidParameter`].
    pub fn parse(s: &str) -> Result<Device> {
        let lower = s.trim().to_ascii_lowercase();
        let want = match lower.as_str() {
            "auto" => return Ok(Device::auto()),
            "cpu" => return Ok(Device::Cpu),
            "cuda" | "gpu" => 0usize,
            other => match other.strip_prefix("cuda:").or_else(|| other.strip_prefix("gpu:")).and_then(|n| n.parse::<usize>().ok()) {
                Some(n) => n,
                None => {
                    return Err(Error::InvalidParameter(format!(
                        "device must be \"auto\", \"cpu\", \"cuda\" or \"cuda:N\", got {s:?}"
                    )))
                }
            },
        };
        if !cuda_built() {
            return Err(Error::gpu("scfast was built without CUDA support (feature `cuda`); use device = \"cpu\" or \"auto\""));
        }
        let info = device_info();
        if !info.cuda_available {
            return Err(Error::gpu(format!(
                "no usable CUDA device: {}",
                info.cuda_error.as_deref().unwrap_or("driver loaded but no device is visible")
            )));
        }
        if want >= info.device_count {
            return Err(Error::gpu(format!("CUDA device {want} requested but only {} visible", info.device_count)));
        }
        Ok(Device::Cuda(want))
    }

    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::Cuda(_))
    }

    /// `"cpu"` or `"cuda:N"`.
    pub fn name(&self) -> String {
        match self {
            Device::Cpu => "cpu".to_string(),
            Device::Cuda(i) => format!("cuda:{i}"),
        }
    }
}

impl Default for Device {
    fn default() -> Self {
        Device::Cpu
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name())
    }
}

/// What the bindings report in `scfast_info()` / `scfast.device_info()`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeviceInfo {
    /// Compiled with feature `cuda`.
    pub cuda_built: bool,
    /// The driver and NVRTC libraries loaded and at least one device is visible.
    pub cuda_available: bool,
    /// Number of visible CUDA devices (0 when unavailable).
    pub device_count: usize,
    /// Name of device 0, when available.
    pub device_name: Option<String>,
    /// Free / total memory of device 0 in bytes, when available.
    pub free_bytes: Option<usize>,
    pub total_bytes: Option<usize>,
    /// Compute capability of device 0, when available.
    pub compute_capability: Option<(i32, i32)>,
    /// Why CUDA is unavailable (library not found, driver error), when it is not.
    pub cuda_error: Option<String>,
}

impl DeviceInfo {
    /// Free memory of device 0 in GiB, for human-readable reports.
    pub fn free_gb(&self) -> Option<f64> {
        self.free_bytes.map(|b| b as f64 / (1u64 << 30) as f64)
    }
}

/// Whether the crate was compiled with the `cuda` feature.
pub const fn cuda_built() -> bool {
    cfg!(feature = "cuda")
}

/// Whether a CUDA device can be used right now (cheap after the first call: the probe
/// result is cached for the life of the process).
pub fn cuda_available() -> bool {
    device_info().cuda_available
}

/// Probe the CUDA runtime environment. Never fails: when the libraries are missing or the
/// driver reports an error, `cuda_available` is false and `cuda_error` says why.
pub fn device_info() -> DeviceInfo {
    #[cfg(feature = "cuda")]
    {
        crate::cuda::probe().clone()
    }
    #[cfg(not(feature = "cuda"))]
    {
        DeviceInfo { cuda_built: false, cuda_error: Some("scfast built without the `cuda` feature".into()), ..Default::default() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cpu_and_auto() {
        assert_eq!(Device::parse("cpu").unwrap(), Device::Cpu);
        assert_eq!(Device::parse(" CPU ").unwrap(), Device::Cpu);
        // auto never fails; on this machine (no GPU) it is the CPU.
        let auto = Device::parse("auto").unwrap();
        assert_eq!(auto, Device::auto());
        if !cuda_available() {
            assert_eq!(auto, Device::Cpu);
        }
    }

    #[test]
    fn parse_rejects_unknown() {
        for s in ["tpu", "", "cuda:x", "gpu:-1", "cuda0"] {
            match Device::parse(s) {
                Err(Error::InvalidParameter(m)) => assert!(m.contains("device"), "{m}"),
                other => panic!("{s:?}: {other:?}"),
            }
        }
    }

    #[test]
    fn cuda_request_without_hardware_is_a_gpu_error() {
        if cuda_available() {
            return; // exercised by the ignored GPU tests instead
        }
        for s in ["cuda", "gpu", "cuda:0", "CUDA:3"] {
            match Device::parse(s) {
                Err(Error::Gpu(m)) => {
                    if cuda_built() {
                        assert!(m.contains("no usable CUDA device"), "{m}");
                    } else {
                        assert!(m.contains("without CUDA support"), "{m}");
                    }
                }
                other => panic!("{s:?}: {other:?}"),
            }
        }
    }

    #[test]
    fn info_is_consistent() {
        let info = device_info();
        assert_eq!(info.cuda_built, cuda_built());
        if !info.cuda_built {
            assert!(!info.cuda_available);
            assert_eq!(info.device_count, 0);
            assert!(info.cuda_error.is_some());
        }
        if !info.cuda_available {
            assert_eq!(Device::auto(), Device::Cpu);
            assert!(info.device_name.is_none());
            assert!(info.cuda_error.is_some(), "an unavailable CUDA must explain itself");
        } else {
            assert!(info.device_count >= 1);
            assert_eq!(Device::auto(), Device::Cuda(0));
        }
        assert_eq!(Device::Cpu.name(), "cpu");
        assert_eq!(Device::Cuda(2).name(), "cuda:2");
        assert_eq!(Device::default(), Device::Cpu);
    }
}
