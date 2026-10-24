// R loads scfastR.so and calls R_init_scfastR. Forward it to the routine registration that
// extendr_module! generates in the Rust static library (src/rust/src/lib.rs); referencing it
// here also stops the linker from dropping the static library.

void R_init_scfastR_extendr(void *dll);

void R_init_scfastR(void *dll) {
    R_init_scfastR_extendr(dll);
}
