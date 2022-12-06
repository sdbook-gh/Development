extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn;

#[proc_macro_derive(MyTrait)] // declared to support #[derive(MyTrait)]
pub fn add_MyTrait(input: TokenStream) -> TokenStream {
    // 基于 input 构建 AST 语法树
    let ast: syn::DeriveInput = syn::parse(input).unwrap();
    // 构建特征实现代码
    impl_MyTrait(&ast)
}

fn impl_MyTrait(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let gen = quote! {
        impl MyTrait for #name {
            fn do_test() {
                println!("{} do_test!", stringify!(#name));
            }
        }
    };
    gen.into()
}
