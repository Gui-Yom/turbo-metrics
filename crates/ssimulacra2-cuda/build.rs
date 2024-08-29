use std::env;

fn main() {
    let out_dir = &env::var("OUT_DIR").expect("can read OUT_DIR");
}
