use vergen::{ConstantsFlags, generate_cargo_keys};

fn main() {
    let mut flags = ConstantsFlags::all();
    flags.toggle(ConstantsFlags::SEMVER_FROM_CARGO_PKG);
    generate_cargo_keys(flags).expect("unable to generate cargo keys!");
}
