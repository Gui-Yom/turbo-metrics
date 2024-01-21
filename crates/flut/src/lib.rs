use proc_macro::{TokenStream, TokenTree};

#[proc_macro]
pub fn flut(stream: TokenStream) -> TokenStream {
    let mut iter = stream.into_iter();
    let TokenTree::Ident(name) = iter.next().unwrap() else {
        panic!()
    };
    let TokenTree::Group(group1) = iter.next().unwrap() else {
        panic!()
    };
    let TokenTree::Group(group2) = iter.next().unwrap() else {
        panic!()
    };
    let items1: Vec<_> = group1
        .stream()
        .into_iter()
        .map(|tt| match tt {
            TokenTree::Group(_) => {
                unreachable!()
            }
            TokenTree::Ident(_) => {
                unreachable!()
            }
            TokenTree::Punct(_) => {
                unreachable!()
            }
            TokenTree::Literal(lit) => lit,
        })
        .collect();
    dbg!(items1);
    for tok in iter {
        println!("{tok:#?}");
    }
    TokenStream::default()
}
