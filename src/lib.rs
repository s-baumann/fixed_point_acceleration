#![deny(missing_debug_implementations)]

pub mod i_structs;
pub mod ii_acceleration;
pub mod iii_fun;

// Optional: "Re-exporting" 
// This lets users call `your_crate::fixed_point` 
// instead of `your_crate::iii_fun::fixed_point`
pub use iii_fun::{fixed_point, fixed_point_from};
pub use i_structs::*;