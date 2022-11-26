type GenericError = Box<dyn std::error::Error + Send + Sync + 'static>;
type GenericResult<T> = Result<T, GenericError>;

fn do_test_sub(val: i32) -> Result<i32, String> {
    let val1 = match std::fs::File::create("test") {
        Ok(v) => v,
        Err(e) => return Err(e.to_string()),
    };
    Ok(0)
}
fn do_test(val: i32) -> Result<i32, String> {
    do_test_sub(val)?;
    Ok(100)
}
fn do_test_sub_generic(pattern: &str) -> GenericResult<i32> {
    let regex_handler = regex::Regex::new(pattern)?;
    Ok(0)
}
fn do_test_generic1(pattern: &str) -> i32 {
    if let Some(err_result) = do_test_sub_generic("").as_ref().err() {
        eprintln!("do_test_sub_generic error {:?}", err_result);
        return -1;
    }
    0
}
fn do_test_generic2(pattern: &str) -> GenericResult<i32> {
    if let Some(err_result) = do_test_sub_generic("").err() {
        eprintln!("do_test_sub_generic error {:?}", &err_result);
        return Err(err_result);
    }
    Ok(0)
}
fn print_vec<T: std::fmt::Display>(n: &[T]) {
    for elt in n {
        println!("-- {} --", elt);
    }
}
fn test_ownership(strval: String) {
    println!("{}", strval);
}

static GLOBAL_VAR: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
lazy_static::lazy_static! {
    static ref HOSTNAME: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());
}

fn main() {
    // {
    //     trait TypeInfo {
    //         fn type_name() -> String;
    //         fn type_of(&self) -> String;
    //     }
    //     macro_rules! impl_type_info {
    //         ($($name:ident$(<$($T:ident),+>)*),*) => {
    //             $(impl_type_info_single!($name$(<$($T),*>)*);)*
    //         };
    //     }
    //     macro_rules! mut_if {
    //         ($name:ident = $value:expr, $($any:expr)+) => {
    //             let mut $name = $value;
    //         };
    //         ($name:ident = $value:expr,) => {
    //             let $name = $value;
    //         };
    //     }
    //     macro_rules! impl_type_info_single {
    //         ($name:ident$(<$($T:ident),+>)*) => {
    //             impl$(<$($T: TypeInfo),*>)* TypeInfo for $name$(<$($T),*>)* {
    //                 fn type_name() -> String {
    //                     mut_if!(res = String::from(stringify!($name)), $($($T)*)*);
    //                     $(
    //                         res.push('<');
    //                         $(
    //                             res.push_str(&$T::type_name());
    //                             res.push(',');
    //                         )*
    //                         res.pop();
    //                         res.push('>');
    //                     )*
    //                     res
    //                 }
    //                 fn type_of(&self) -> String {
    //                     $name$(::<$($T),*>)*::type_name()
    //                 }
    //             }
    //         }
    //     }
    //     impl<'a, T: TypeInfo + ?Sized> TypeInfo for &'a T {
    //         fn type_name() -> String {
    //             let mut res = String::from("&");
    //             res.push_str(&T::type_name());
    //             res
    //         }
    //         fn type_of(&self) -> String {
    //             <&T>::type_name()
    //         }
    //     }
    //     impl<'a, T: TypeInfo + ?Sized> TypeInfo for &'a mut T {
    //         fn type_name() -> String {
    //             let mut res = String::from("&mut ");
    //             res.push_str(&T::type_name());
    //             res
    //         }
    //         fn type_of(&self) -> String {
    //             <&mut T>::type_name()
    //         }
    //     }
    //     macro_rules! type_of {
    //         ($x:expr) => {
    //             (&$x).type_of()
    //         };
    //     }

    //     impl_type_info!(i32, i64, f32, f64, str, String, Vec<T>, Result<T,S>);
    //     println!("{}", type_of!(1));
    //     println!("{}", type_of!(&1));
    //     println!("{}", type_of!(&&1));
    //     println!("{}", type_of!(&mut 1));
    //     println!("{}", type_of!(&&mut 1));
    //     println!("{}", type_of!(&mut &1));
    //     println!("{}", type_of!(1.0));
    //     println!("{}", type_of!("abc"));
    //     println!("{}", type_of!(&"abc"));
    //     println!("{}", type_of!(String::from("abc")));
    //     println!("{}", type_of!(vec![1, 2, 3]));
    //     println!("{}", <Result<String, i64>>::type_name());
    //     println!("{}", <&i32>::type_name());
    //     println!("{}", <&str>::type_name());
    // }
    // return;
    {
        macro_rules! my_vec {
            ( $elem:tt size $n:tt) => { // use tt to use word seperate pattern
            ::std::vec::from_elem($elem, $n)
            };
            ( $( $x:expr ) ; *) => { // only support ',' ' ' ';' '=>'
                {
                    let mut v = ::std::vec::Vec::new();
                    $( v.push($x); )*
                    v
                }
            };
            ( $( $x:tt ) ; + first $y:tt) => { // use word first as seperate pattern
                {
                    let mut v = ::std::vec::Vec::new();
                    $( v.push($x); )*
                    v[0] = $y;
                    v
                }
            }
        }
        let val = my_vec!(1 size 1);
        dbg!(val);
        let val = my_vec!(1 ; 1);
        dbg!(val);
        let val = my_vec!(1 ; 1 first 50);
        dbg!(val);

        macro_rules! rep {
            () => {-1};
            ($ ($e:expr) ,+) => {
                {
                    let mut v = Vec::new();
                    $(
                        v.push($e);
                    )+
                    v
                }
            };
            ($ ($e:expr) ,+ ,) => {
                {
                    // rep![ $( $e ),+ ]
                    let mut sum = 0;
                    $(
                        sum = sum + $e;
                    )+
                    sum
                }
            };
        }
        println!("{:?}", rep![]);
        println!("{:?}", rep![1, 2, 3]);
        println!("{:?}", rep![1, 2, 3,]);

        macro_rules! param_count {
            ($a:tt + $b:tt) => {
                "got an a+b expression"
            };
            ($i:ident) => {
                "got an identifier"
            };
            ($a:tt kiss $b:tt) => {
                $a + $b
            };
            ($($e:tt)*) => {
                "got some tokens"
            };
        }
        println!("{}", param_count!(3 + 4));
        println!("{}", param_count!(a));
        println!("{}", param_count!(7 kiss 8));
        println!("{}", param_count!(4 5 6));

        macro_rules! my_cc {
            (
                struct $name:ident {$(pub $field_name:ident: $field_type:ty,)*}
            )
         => {
                struct $name {
                    $($field_name: $field_type,)*
                }
                impl $name {
                    fn log(&self) {
                        $( println!("{} -> {:?}", stringify!($field_name), self.$field_name); )*
                    }
                }
            }
        }

        my_cc!(
            struct Hello {
                pub name: String,
                pub size: String,
            }
        );
        let hello = Hello {
            name: "1".to_string(),
            size: "2".to_string(),
        };
        hello.log();

        pub trait HelloMacro {
            fn hello_macro();
        }
        use hello_macro_derive::HelloMacro;
        #[derive(HelloMacro)]
        struct MyStruct;
        MyStruct::hello_macro();

        {
            use std::collections::HashMap;
            #[derive(Clone, PartialEq, Debug)]
            enum Json {
                Null,
                Boolean(bool),
                Number(f64),
                String(String),
                Array(Vec<Json>),
                Object(Box<HashMap<String, Json>>),
            }
            impl From<bool> for Json {
                fn from(b: bool) -> Json {
                    Json::Boolean(b)
                }
            }
            impl From<String> for Json {
                fn from(s: String) -> Json {
                    Json::String(s)
                }
            }
            impl<'a> From<&'a str> for Json {
                fn from(s: &'a str) -> Json {
                    Json::String(s.to_string())
                }
            }
            macro_rules! impl_from_num_for_json {
                ( $( $t:ident )* ) => {
                $(
                impl From<$t> for Json {
                    fn from(n: $t) -> Json {
                        Json::Number(n as f64)
                    }
                }
                )*
                };
            }
            impl_from_num_for_json!(u8 i8 u16 i16 u32 i32 u64 i64 u128 i128 usize isize f32 f64);

            macro_rules! json {
                (null) => {
                    Json::Null
                };
                ([ $( $element:tt),* ]) => {
                    Json::Array(vec![ $( json!($element) ),* ])
                };
                ({ $( $key:tt : $value:tt ),* }) => {
                    Json::Object(Box::new(vec![
                    $( ($key.to_string(), json!($value)) ),*
                    ].into_iter().collect()))
                };
                ( $other:tt ) => {
                    Json::from($other) // 处理布尔/数字/字符串
                };
            }
            assert_eq!(json!(null), Json::Null); // passes!
            let width = 4.0;
            let desc = json!({
                "width": 1,
                "height": 2
            });
            dbg!(desc);
        }
    }
    return;
    {
        fn generic_str_fn<'a>() -> &'a str {
            "str"
        }
        fn static_str_fn() -> &'static str {
            "str"
        }
        fn a_or_b<T>(a: T, b: T) -> T {
            if rand::random() {
                a
            } else {
                b
            }
        }

        let some_string = "string".to_owned();
        let some_str = &some_string[..];
        let str_ref = a_or_b(some_str, generic_str_fn()); // compiles
        let str_ref = a_or_b(some_str, static_str_fn()); // compiles
    }
    return;
    {
        let mut identity: Box<dyn Fn(&i32) -> &i32> = Box::new(|x: &i32| x);
        identity = Box::new(|x: &i32| &1234);
        dbg!(identity(&0));
        let mut identity: &dyn Fn(i32) -> i32 = &|x: i32| x;
        identity = &|x: i32| x * 100;
        dbg!(identity(1));
        struct MyStruct1;
        struct MyStruct2;
        type CallBack<'a> = &'a dyn Fn(&mut MyStruct1, MyStruct2) -> bool;
        let func: CallBack;
        let mut e1 = MyStruct1;
        let e2 = MyStruct2;
        func = &|e1, e2| false;
        dbg!(func(&mut e1, e2));
        type CallBackRef<'a> = std::cell::Ref<'a, dyn Fn(&mut MyStruct1, MyStruct2) -> bool>;

        struct Todo {
            id: i16,
            deleted: bool,
            completed: bool,
        }
        fn with_todo_id_1(todos: &mut Vec<Todo>, todo_id: i16, f: &dyn Fn(&mut Todo)) {
            if let Some(todo) = todos.iter_mut().find(|todo| todo.id == todo_id) {
                f(todo);
            }
        }
        fn with_todo_id_2<P>(todos: &mut Vec<Todo>, todo_id: i16, f: P)
        where
            P: Fn(&mut Todo),
        {
            if let Some(todo) = todos.iter_mut().find(|todo| todo.id == todo_id) {
                f(todo);
            }
        }
        fn remove_todo(todos: &mut Vec<Todo>, todo_id: i16) {
            with_todo_id_1(todos, todo_id, &|todo| todo.deleted = true);
        }
        fn mark_done(todos: &mut Vec<Todo>, todo_id: i16) {
            with_todo_id_2(todos, todo_id, |todo| todo.completed = true);
        }

        {
            // iter and closure sample
            struct MyFromFn<F>(F);
            fn my_from_fn<T, F>(f: F) -> MyFromFn<F>
            where
                F: FnMut() -> Option<T>,
            {
                MyFromFn(f)
            }
            impl<T, F> Iterator for MyFromFn<F>
            where
                F: FnMut() -> Option<T>,
            {
                type Item = T;
                fn next(&mut self) -> Option<Self::Item> {
                    (self.0)()
                }
            }
            fn fibonacci() -> impl Iterator<Item = usize> {
                let mut state = (0, 1);
                // std::iter::from_fn(move || { // create iter with function callback
                my_from_fn(move || {
                    // move to mark ref outer var
                    state = (state.1, state.0 + state.1);
                    Some(state.0)
                })
            }
            let result = fibonacci().nth(4).unwrap();
            dbg!(result);
        }
    }
    return;
    {
        struct Struct {
            mutex: std::sync::Mutex<String>,
        }
        impl Struct {
            // downgrades mut self to shared str
            fn get_string(&mut self) -> &str {
                self.mutex.get_mut().unwrap() // mutex.get_mut() borrows the Mutex mutably, no actual locking needs to take place
            }
            fn mutate_string(&self) {
                // if Rust allowed downgrading mut refs to shared refs
                // then the following line would invalidate any shared
                // refs returned from the get_string method
                let mut val = self.mutex.lock().unwrap();
                *val = "surprise!".to_owned();
            }
        }
        let mut s = Struct {
            mutex: std::sync::Mutex::new("string".to_owned()),
        };
        let str_ref = s.get_string(); // mut ref downgraded to shared ref
        s.mutate_string(); // str_ref invalidated, now a dangling pointer
                           // dbg!(str_ref); // compile error as expected

        #[derive(Debug, Default)]
        struct Player;
        let mut server: std::collections::HashMap<String, Player> =
            std::collections::HashMap::new();
        let player_a = "1".to_string();
        let player_b = "2".to_string();
        // get players from server or create & insert new players if they don't yet exist
        let player_a: &Player = server.entry(player_a).or_default();
        let player_b: &Player = server.entry(player_b).or_default();
        // dbg!(player_a, player_b); // compile error
    }
    return;
    {
        struct ByteIter<'a> {
            remainder: &'a [u8],
        }
        impl<'a> ByteIter<'a> {
            fn modify1(&mut self) -> Option<&u8> {
                Some(&1)
            }
            fn modify2(&mut self) -> Option<&u8> {
                Some(&2)
            }
            fn newmodify1(&mut self) -> Option<&'a u8> {
                Some(&1)
            }
            fn newmodify2(&mut self) -> Option<&'a u8> {
                Some(&2)
            }
            fn some_method(&'a mut self) {} // this method will mutably borrow the struct for the entirety of the struct's lifetime
        }
        let mut bytes = ByteIter { remainder: b"1234" };
        // let byte_n1 = bytes.modify1();
        // let byte_n2 = bytes.modify2(); // cannot borrow `bytes` as mutable more than once at a time
        // dbg!(byte_n1);
        // dbg!(byte_n2);
        let byte_n1 = bytes.newmodify1();
        let byte_n2 = bytes.newmodify2();
        dbg!(byte_n1);
        dbg!(byte_n2);
        // bytes.some_method();
        // bytes.some_method(); // cannot borrow `bytes` as mutable more than once at a time

        trait Animal {
            fn walk(&self) {
                println!("walk");
            }
        }
        impl dyn Animal {
            fn talk(&self) {
                println!("talk");
            }
        }
        struct Person;
        impl Animal for Person {}
        fn demo() -> Box<dyn Animal> {
            Box::new(Person {})
        }
        let p = Person;
        p.walk();
        let p1 = demo();
        p1.talk();

        trait Trait {}
        // elided
        type T1 = Box<dyn Trait>;
        // expanded, Box<T> has no lifetime bound on T, so inferred as 'static
        type T2 = Box<dyn Trait + 'static>;
        // elided
        impl dyn Trait {}
        // expanded
        impl dyn Trait + 'static {}
        // elided
        type T3<'a> = &'a dyn Trait;
        // expanded, &'a T requires T: 'a, so inferred as 'a
        type T4<'a> = &'a (dyn Trait + 'a);
        // elided
        type T5<'a> = std::cell::Ref<'a, dyn Trait>;
        // expanded, Ref<'a, T> requires T: 'a, so inferred as 'a
        type T6<'a> = std::cell::Ref<'a, dyn Trait + 'a>;
        trait GenericTrait<'a>: 'a {}
        // elided
        type T7<'a> = Box<dyn GenericTrait<'a>>;
        // expanded
        type T8<'a> = Box<dyn GenericTrait<'a> + 'a>;
        // elided
        impl<'a> dyn GenericTrait<'a> {}
        // expanded
        impl<'a> dyn GenericTrait<'a> + 'a {}

        let c = std::cell::Cell::new("asdf");
        let one = c.get();
        c.set("qwer");
        let two = c.get();
        println!("{},{}", one, two);
        let str = "我很善变, 还拥有多个主人".to_string();
        let s = std::rc::Rc::new(std::cell::RefCell::new(str));
        let s1 = s.clone();
        let s2 = s.clone();
        s2.borrow_mut().push_str(", on yeah!");
        println!("{:?}\n{:?}\n{:?}", s, s1, s2);
    }
    return;
    {
        pub struct MyFuture<T>(std::sync::Arc<std::sync::Mutex<Shared<T>>>);
        struct Shared<T> {
            value: Option<T>,
            waker: Option<std::task::Waker>,
        }
        pub fn my_spawn<T, F>(closure: F) -> MyFuture<T>
        where
            F: FnOnce() -> T,
            F: Send + 'static,
            T: Send + 'static,
        {
            let inner = std::sync::Arc::new(std::sync::Mutex::new(Shared::<T> {
                value: None,
                waker: None,
            }));
            std::thread::spawn({
                let inner = inner.clone();
                move || {
                    let value = closure();
                    let maybe_waker = {
                        let mut guard = inner.lock().unwrap();
                        guard.value = Some(value);
                        guard.waker.take()
                    };
                    if let Some(waker) = maybe_waker {
                        waker.wake();
                    }
                }
            });
            MyFuture(inner)
        }
        impl<T: Send> std::future::Future for MyFuture<T> {
            type Output = T;
            fn poll(
                self: std::pin::Pin<&mut Self>,
                cx: &mut std::task::Context<'_>,
            ) -> std::task::Poll<Self::Output> {
                let mut guard = self.0.lock().unwrap();
                if let Some(value) = guard.value.take() {
                    return std::task::Poll::Ready(value);
                }
                guard.waker = Some(cx.waker().clone());
                std::task::Poll::Pending
            }
        }
        fn my_block_on<F: std::future::Future>(future: F) -> F::Output {
            let parker = crossbeam::sync::Parker::new();
            let unparker = parker.unparker().clone();
            let waker = waker_fn::waker_fn(move || unparker.unpark());
            let mut context = std::task::Context::from_waker(&waker);
            futures_lite::pin!(future);
            loop {
                match future.as_mut().poll(&mut context) {
                    std::task::Poll::Ready(value) => return value,
                    std::task::Poll::Pending => parker.park(),
                }
            }
        }

        {
            // Pin
            struct Test {
                a: String,
                b: *const String,
                _pinned: std::marker::PhantomPinned, // mark to !Unpin
            }
            impl Test {
                fn new(txt: &str) -> std::pin::Pin<Box<Self>> {
                    let mut test = Box::pin(Test {
                        a: String::from(txt),
                        b: std::ptr::null(),
                        _pinned: std::marker::PhantomPinned, // mark to !Unpin
                    });
                    unsafe {
                        test.as_mut().get_unchecked_mut().b = &(test.a);
                    }
                    test
                }
                fn a<'a>(self: &'a std::pin::Pin<Box<Self>>) -> &'a str {
                    &(self.as_ref().get_ref().a)
                }
                fn b<'a>(self: &'a std::pin::Pin<Box<Self>>) -> &'a String {
                    unsafe { &*self.as_ref().get_ref().b }
                }
            }
            let mut test1 = Test::new("test1");
            let mut test2 = Test::new("test2");
            println!("a: {}, b: {}", test1.a(), test1.b());
            println!("a: {}, b: {}", test2.a(), test2.b());
        };
    }
    return;
    {
        async fn test_rc(t: std::rc::Rc<String>) {
            println!("{:?}", std::thread::current().id());
            let k = t;
            println!("1. test,{:?}", k);
            async_std::task::sleep(std::time::Duration::from_secs(1)).await;
            println!("2. test,{:?}", k);
        }
        async fn test_arc(t: std::sync::Arc<String>) {
            println!("{:?}", std::thread::current().id());
            let k = t;
            println!("1. test,{:?}", k);
            async_std::task::sleep(std::time::Duration::from_secs(1)).await;
            println!("2. test,{:?}", k);
        }
        let r = std::rc::Rc::new("abc".to_string());
        let f1 = test_rc(r.clone());
        let f2 = test_rc(r.clone());
        let f3 = test_rc(r.clone());
        let f4 = test_rc(r.clone());
        use async_std::prelude::*;
        // async_std::task::block_on(f1.race(f2).race(f3).race(f4)); // same thread, so Rc can work
        let r = std::sync::Arc::new("abc".to_string());
        let f1 = test_arc(r.clone());
        let f2 = test_arc(r.clone());
        async_std::task::spawn(f1);
        async_std::task::spawn(f2);
        async_std::task::block_on(async {
            println!("sleep");
            async_std::task::sleep(std::time::Duration::from_secs(5)).await; // make block_on live long enough
        });
    }
    return;
    if false {
        // High Rank Trait Bound
        struct Closure<F> {
            data1: u8,
            data2: u16,
            func: F,
        }
        impl<F> Closure<F>
        where
            F: for<'b> Fn(&u8, &'b u16) -> &'b u16,
        {
            fn call_new(&self) -> &u16 {
                (self.func)(&self.data1, &self.data2)
            }
        }
        fn do_it<'b>(data1: &u8, data2: &'b u16) -> &'b u16 {
            &data2
        }
        let clo = Closure {
            data1: 0u8,
            data2: 1u16,
            func: do_it,
        };
        println!("{}", clo.call_new());
    }
    return;
    {
        use std::io::prelude::*;
        fn simple_request(host: &str, port: u16, path: &str) -> std::io::Result<String> {
            let mut socket = std::net::TcpStream::connect((host, port))?;
            let request = format!("GET {} HTTP/1.1\r\nHost: {}\r\n\r\n", path, host);
            socket.write_all(request.as_bytes())?;
            socket.shutdown(std::net::Shutdown::Write)?;
            let mut response = String::new();
            socket.read_to_string(&mut response)?;
            Ok(response)
        }
        let response = simple_request("www.baidu.com", 80, "/index.html").expect("network error");
        // println!("sync response\n{}", response);

        use async_std::io::prelude::*;
        async fn simple_request_async(
            host: &str,
            port: u16,
            path: &str,
        ) -> std::io::Result<String> {
            println!("start request in {:?}", std::thread::current().id());
            println!("connect");
            let mut socket = async_std::net::TcpStream::connect((host, port)).await?;
            let request = format!("GET {} HTTP/1.1\r\nHost: {}\r\n\r\n", path, host);
            println!("write_all");
            socket.write_all(request.as_bytes()).await?;
            println!("shutdown");
            socket.shutdown(async_std::net::Shutdown::Write)?;
            let mut response = String::new();
            println!("read_to_string");
            socket.read_to_string(&mut response).await?;
            println!("ok");
            Ok(response)
        }
        let response =
            async_std::task::block_on(simple_request_async("www.baidu.com", 80, "/index.html"))
                .expect("network error");
        // println!("async response\n{}", response);

        async fn multi_simple_requests(
            requests: Vec<(String, u16, String)>,
        ) -> Vec<std::io::Result<String>> {
            let mut handles = vec![];
            for (host, port, path) in requests {
                // handles.push(async_std::task::spawn(async move {
                //     simple_request_async(&host, port, &path).await
                // }));
                handles.push(async_std::task::spawn_local(async move {
                    simple_request_async(&host, port, &path).await
                }));
            }
            let mut results = vec![];
            for handle in handles {
                results.push(handle.await);
            }
            results
        }
        let results = async_std::task::block_on(multi_simple_requests(vec![
            ("www.baidu.com".to_string(), 80, "/index.html".to_string()),
            ("www.sohu.com".to_string(), 80, "/index.html".to_string()),
            ("www.bing.com".to_string(), 80, "/index.html".to_string()),
        ]));
        results.iter().for_each(|x| match x {
            Err(ref e) => println!("network error {}", e.to_string()),
            Ok(ref r) => println!("result {:.100}", r),
        });

        let input = async_std::io::stdin();
        let future = async {
            println!("Please input line");
            let mut line = String::new();
            input.read_line(&mut line).await?;
            println!("Read line: {}", line);
            Ok::<(), std::io::Error>(())
        };
        // async_std::task::block_on(future).unwrap();

        fn simple_request_none_async<'a>(
            host: &'a str,
            port: u16,
            path: &'a str,
        ) -> impl std::future::Future<Output = std::io::Result<String>> + 'a {
            async move {
                let mut socket = async_std::net::TcpStream::connect((host, port)).await?;
                let request = format!("GET {} HTTP/1.1\r\nHost: {}\r\n\r\n", path, host);
                socket.write_all(request.as_bytes()).await?;
                socket.shutdown(async_std::net::Shutdown::Write)?;
                let mut response = String::new();
                socket.read_to_string(&mut response).await?;
                Ok(response)
            }
        }
        fn simple_request_none_async_move(
            host: &str,
            port: u16,
            path: &str,
        ) -> impl std::future::Future<Output = std::io::Result<String>> + 'static {
            let host = host.to_string();
            let path = path.to_string();
            async move {
                let mut socket = async_std::net::TcpStream::connect((host.as_str(), port)).await?;
                let request = format!("GET {} HTTP/1.1\r\nHost: {}\r\n\r\n", path, host);
                socket.write_all(request.as_bytes()).await?;
                socket.shutdown(async_std::net::Shutdown::Write)?;
                let mut response = String::new();
                socket.read_to_string(&mut response).await?;
                Ok(response)
            }
        }

        if false {
            async fn async_test_func() {}
            async fn reluctant() -> String {
                let string = std::rc::Rc::new("ref-counted String".to_string());
                async_test_func().await;
                format!("Your splendid string: {}", string)
            }
            // async_std::task::spawn(reluctant()); // Rc cannot satisfy send
            type GenericError = Box<dyn std::error::Error>;
            type GenericResult<T> = Result<T, GenericError>;
            fn some_fallible_thing() -> GenericResult<i32> {
                Ok(0)
            }
            async fn unfortunate() {
                // ...因为这个调用返回的值...
                match some_fallible_thing() {
                    Err(error) => {
                        // report_error(error);
                    }
                    Ok(output) => {
                        // ...到这个 await处仍然存在...
                        async_test_func().await;
                    }
                }
            }
            // async_std::task::spawn(unfortunate()); // Box<dyn std::error::Error> cannot satisfy send
        }

        if false {
            async fn async_run_func() {
                println!(
                    "async_run_func start {} {:?}",
                    chrono::Utc::now().to_string(),
                    std::thread::current().id()
                );
                std::thread::sleep(std::time::Duration::from_secs(5));
                println!(
                    "async_run_func finish {} {:?}",
                    chrono::Utc::now().to_string(),
                    std::thread::current().id()
                );
            }
            let future = async_run_func();
            std::thread::sleep(std::time::Duration::from_secs(2));
            println!(
                "after future created {} {:?}",
                chrono::Utc::now().to_string(),
                std::thread::current().id()
            );
            let result = async_std::task::block_on(future);
            println!(
                "after future await {} {:?}",
                chrono::Utc::now().to_string(),
                std::thread::current().id()
            );

            fn sync_run_func() {
                println!(
                    "sync_run_func start {} {:?}",
                    chrono::Utc::now().to_string(),
                    std::thread::current().id()
                );
                std::thread::sleep(std::time::Duration::from_secs(5));
                println!(
                    "sync_run_func finish {} {:?}",
                    chrono::Utc::now().to_string(),
                    std::thread::current().id()
                );
            }
            let future = async_std::task::spawn_blocking(move || {
                println!(
                    "spawn_blocking start {} {:?}",
                    chrono::Utc::now().to_string(),
                    std::thread::current().id()
                );
                sync_run_func();
                println!(
                    "spawn_blocking finish {} {:?}",
                    chrono::Utc::now().to_string(),
                    std::thread::current().id()
                );
            });
            std::thread::sleep(std::time::Duration::from_secs(2));
            println!(
                "after future created {} {:?}",
                chrono::Utc::now().to_string(),
                std::thread::current().id()
            );
            let result = async_std::task::block_on(future);
            println!(
                "after future await {} {:?}",
                chrono::Utc::now().to_string(),
                std::thread::current().id()
            );
        }

        async fn multi_surf_requests(
            urls: &[&str],
        ) -> std::vec::Vec<Result<String, surf::Exception>> {
            let client = surf::Client::new();
            let mut handles = vec![];
            for url in urls {
                let request = client.get(url).recv_string();
                handles.push(async_std::task::spawn(request));
            }
            let mut results = vec![];
            for handle in handles {
                results.push(handle.await);
            }
            results
        }
        let requests = [
            "http://example.com",
            "https://www.red-bean.com",
            "https://en.wikipedia.org/wiki/Main_Page",
        ];
        async_std::task::block_on(multi_surf_requests(&requests))
            .iter()
            .enumerate()
            .for_each(|(idx, r)| match r {
                Err(e) => println!("{} -- {}", idx, e.to_string()),
                Ok(s) => println!("{} -- {:.100}", idx, s),
            });

        if std::env::args().nth(1) == Some("chatclient".to_string()) {
            use async_std::prelude::*;
            let address = std::env::args()
                .nth(2)
                .expect("Usage: chatclient ADDRESS:PORT");
            async_std::task::block_on(async {
                let socket = async_std::net::TcpStream::connect(address)
                    .await
                    .expect("connect error");
                socket.set_nodelay(true).expect("set_nodelay error");
                let to_server = async_chat::send_commands(socket.clone());
                let from_server = async_chat::handle_replies(socket);
                from_server
                    .race(to_server)
                    .await
                    .expect("during chat error");
            });
        } else if std::env::args().nth(1) == Some("chatserver".to_string()) {
            use async_std::prelude::*;
            let address = std::env::args()
                .nth(2)
                .expect("Usage: chatserver ADDRESS:PORT");
            async_std::task::block_on(async {
                let listener = async_std::net::TcpListener::bind(address)
                    .await
                    .expect("bind error");
                let chat_group_table = std::sync::Arc::new(async_chat::GroupTable::new());
                while let Some(socket_result) = listener.incoming().next().await {
                    let socket = socket_result.expect("get incoming socket error");
                    let group_table = chat_group_table.clone();
                    async_std::task::spawn(async {
                        match async_chat::serve(socket, group_table).await {
                            Err(e) => {
                                println!("serve error {}", e.to_string())
                            }
                            _ => (),
                        }
                    });
                }
            });
        }
    }
    return;
    {
        fn test_callback<F>(str: &str, f: F) -> u32
        where
            F: Fn(&str) -> u32,
        {
            f(str) * 100
        }
        let val = test_callback("123", |x| u32::from_str_radix(x, 10).unwrap());
        dbg!(val);
        type callback_t = Box<dyn Fn(&mut String, String) -> bool>;
        fn test_callback_new(f1: callback_t, f2: callback_t) -> bool {
            let mut str1 = String::new();
            let str2 = "_str1".to_string();
            let str3 = "_str2".to_string();
            f1(&mut str1, str2) && f2(&mut str1, str3)
        }
        let mut mut_str = String::new();
        let str = "test".to_string();
        test_callback_new(
            Box::new(|str1, str2| {
                str1.push_str(&str2);
                true
            }),
            Box::new(|str1, str2| {
                str1.push_str(&(str2.to_uppercase()));
                true
            }),
        );
    }
    return;
    {
        pub fn spawn<F, T>(f: F) -> T
        where
            F: FnOnce() -> T,
            F: Send + 'static,
            T: Send + 'static,
        {
            f()
        }

        let val = vec![1].into_iter();
        let result = spawn(move || for item in val {});
        dbg!(result);

        struct MyStruct(std::collections::LinkedList<i32>);
        impl MyStruct {
            fn new() -> MyStruct {
                MyStruct(std::collections::LinkedList::<i32>::new())
            }
            fn in_first(&mut self, val: i32) {
                self.0.push_front(val);
            }
            fn out_all(&mut self) -> std::vec::Vec<i32> {
                let mut vec = std::vec::Vec::<i32>::new();
                for i in 0..self.0.len() {
                    vec.push(self.0.pop_back().unwrap());
                }
                vec
            }
        }
        let mut val = MyStruct::new();
        val.in_first(1);
        val.in_first(10);
        val.in_first(100);
        val.in_first(1000);
        let result = std::thread::spawn(move || {
            for item in val.out_all() {
                println!("{}", item)
            }
        });
        result.join().unwrap();

        {
            // https://course.rs/too-many-lists/intro.html
            pub struct List<T> {
                head: Link<T>,
            }
            type Link<T> = Option<Box<Node<T>>>;
            struct Node<T> {
                elem: T,
                next: Link<T>,
            }
            impl<T> List<T> {
                pub fn new() -> Self {
                    List { head: None }
                }
                pub fn push(&mut self, elem: T) {
                    let new_node = Box::new(Node {
                        elem: elem,
                        next: self.head.take(),
                    });
                    self.head = Some(new_node);
                }
                pub fn pop(&mut self) -> Option<T> {
                    self.head.take().map(|node| {
                        self.head = node.next;
                        node.elem
                    })
                }
            }
            impl<T> Drop for List<T> {
                fn drop(&mut self) {
                    let mut cur_link = self.head.take();
                    while let Some(mut boxed_node) = cur_link {
                        cur_link = boxed_node.next.take();
                    }
                }
            }
            pub struct IntoIter<T>(List<T>);
            impl<T> List<T> {
                pub fn into_iter(self) -> IntoIter<T> {
                    IntoIter(self)
                }
            }
            impl<T> Iterator for IntoIter<T> {
                type Item = T;
                fn next(&mut self) -> Option<Self::Item> {
                    self.0.pop()
                }
            }
            let mut list: List<i32> = List::new();
            list.push(1);
            list.push(3);
            list.push(5);
            list.push(7);
            let val = list.into_iter();
            let result = std::thread::spawn(move || {
                for item in val {
                    println!("{}", item)
                }
            });
            result.join().unwrap();
        }

        pub trait ThreadIterator: std::iter::Iterator {
            fn spawn_thread(self) -> std::sync::mpsc::IntoIter<Self::Item>;
        }
        impl<T> ThreadIterator for T
        where
            T: std::iter::Iterator + std::marker::Send + 'static,
            T::Item: std::marker::Send + std::fmt::Display,
        {
            fn spawn_thread(self) -> std::sync::mpsc::IntoIter<Self::Item> {
                let (sender, receiver) = std::sync::mpsc::sync_channel(100);
                spawn(move || {
                    for item in self {
                        sender.send(item).unwrap();
                    }
                });
                receiver.into_iter()
            }
        }
    }
    return;
    {
        struct MyStruct(u32);
        impl MyStruct {
            fn test_static() -> MyStruct {
                MyStruct(0)
            }
            fn test_method(&self) -> u32 {
                self.0
            }
        }
        let val = MyStruct::test_static();
        let val = val.test_method();
    }
    return;
    {
        struct InMemoryIndex;
        impl InMemoryIndex {
            fn from_single_document(doc_id: usize, text: String) -> InMemoryIndex {
                InMemoryIndex {}
            }
        }
        fn start_file_reader_thread(
            read_file_names: Vec<std::path::PathBuf>,
        ) -> (
            std::sync::mpsc::Receiver<String>,
            std::thread::JoinHandle<std::io::Result<()>>,
        ) {
            let (sender, receiver) = std::sync::mpsc::channel::<String>();
            let handler = std::thread::spawn(move || {
                for item in read_file_names {
                    let content = std::fs::read_to_string(item)?;
                    match sender.send(content) {
                        Err(e) => {
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::Other,
                                e.to_string(),
                            ));
                        }
                        _ => {}
                    }
                }
                Ok(())
            });
            (receiver, handler)
        }
        fn start_file_indexing_thread(
            texts: std::sync::mpsc::Receiver<String>,
        ) -> (
            std::sync::mpsc::Receiver<InMemoryIndex>,
            std::thread::JoinHandle<std::io::Result<()>>,
        ) {
            let (sender, receiver) = std::sync::mpsc::channel();
            let handle = std::thread::spawn(move || {
                for (doc_id, text) in texts.into_iter().enumerate() {
                    let index = InMemoryIndex::from_single_document(doc_id, text);
                    match sender.send(index) {
                        Err(e) => {
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::Other,
                                e.to_string(),
                            ));
                        }
                        _ => {}
                    }
                }
                Ok(())
            });
            (receiver, handle)
        }
        fn start_in_memory_merge_thread(
            file_indexes: std::sync::mpsc::Receiver<InMemoryIndex>,
        ) -> (
            std::sync::mpsc::Receiver<InMemoryIndex>,
            std::thread::JoinHandle<std::io::Result<()>>,
        ) {
            let (sender, receiver) = std::sync::mpsc::channel();
            let handle = std::thread::spawn(move || Ok(()));
            (receiver, handle)
        }
        fn start_index_writer_thread(
            big_indexes: std::sync::mpsc::Receiver<InMemoryIndex>,
            output_dir: &std::path::PathBuf,
        ) -> (
            std::sync::mpsc::Receiver<std::path::PathBuf>,
            std::thread::JoinHandle<std::io::Result<()>>,
        ) {
            let (sender, receiver) = std::sync::mpsc::channel();
            let handle = std::thread::spawn(move || Ok(()));
            (receiver, handle)
        }
        fn merge_index_files(
            files: std::sync::mpsc::Receiver<std::path::PathBuf>,
            output_dir: &std::path::Path,
        ) -> std::io::Result<()> {
            Ok(())
        }
        fn run_pipeline(
            documents: Vec<std::path::PathBuf>,
            output_dir: std::path::PathBuf,
        ) -> std::io::Result<()> {
            let (texts, h1) = start_file_reader_thread(documents);
            let (indecies, h2) = start_file_indexing_thread(texts);
            let (merged_indecies, h3) = start_in_memory_merge_thread(indecies);
            let (files, h4) = start_index_writer_thread(merged_indecies, &output_dir);
            let result = merge_index_files(files, &output_dir);
            let r1 = h1.join().unwrap();
            let r2 = h2.join().unwrap();
            let r3 = h3.join().unwrap();
            let r4 = h4.join().unwrap();
            r1?;
            r2?;
            r3?;
            r4?;
            Ok(())
        }

        let work_vec = vec!["".to_string(); 10];
        let mut thread_handler_vec: Vec<std::thread::JoinHandle<Result<(), &str>>> =
            std::vec::Vec::new();
        let error_exist = false;
        thread_handler_vec.push(std::thread::spawn(move || {
            println!("thread {:?}", std::thread::current().id());
            use chrono::prelude::*;
            for work in work_vec {
                println!("{:?}", Utc::now());
                std::thread::sleep(std::time::Duration::from_secs(1));
                if error_exist {
                    return Err("error msg");
                }
            }
            Ok(())
        }));
        for handler in thread_handler_vec {
            let thread_result: Result<(), &str> = handler.join().unwrap(); // join will return thread's Result
        }

        mod m1 {
            pub trait ThreadIterator: std::iter::Iterator {
                fn spawn_thread(self) -> std::sync::mpsc::IntoIter<Self::Item>;
            }
            impl<T> self::ThreadIterator for T
            where
                T: std::iter::Iterator + std::marker::Send + 'static,
                T::Item: std::marker::Send + std::fmt::Display,
            {
                fn spawn_thread(self) -> std::sync::mpsc::IntoIter<Self::Item> {
                    let (sender, receiver) = std::sync::mpsc::sync_channel(100);
                    std::thread::spawn(move || {
                        for item in self.enumerate() {
                            println!(
                                "{:?} send {} {}",
                                std::thread::current().id(),
                                item.0,
                                item.1
                            );
                            sender.send(item.1).expect("send error");
                        }
                    });
                    receiver.into_iter()
                }
            }
        }
        mod m2 {
            pub trait ThreadIterator: std::iter::Iterator {
                fn spawn_thread(self) -> std::sync::mpsc::IntoIter<Self::Item>;
            }
            impl<T> self::ThreadIterator for T
            where
                T: std::iter::Iterator + std::marker::Send,
                T::Item: std::marker::Send + std::fmt::Display,
            {
                fn spawn_thread(self) -> std::sync::mpsc::IntoIter<Self::Item> {
                    let (sender, receiver) = std::sync::mpsc::sync_channel(100);
                    std::thread::scope(|s| {
                        s.spawn(move || {
                            for item in self.enumerate() {
                                println!(
                                    "{:?} send {} {}",
                                    std::thread::current().id(),
                                    item.0,
                                    item.1
                                );
                                sender.send(item.1).expect("send error");
                            }
                        });
                    });
                    // crossbeam::scope(|spawner| {
                    //     spawner.spawn(move |_| {
                    //         for item in self.enumerate() {
                    //             println!(
                    //                 "{:?} send {} {}",
                    //                 std::thread::current().id(),
                    //                 item.0,
                    //                 item.1
                    //             );
                    //             sender.send(item.1).expect("send error");
                    //         }
                    //     });
                    // });
                    receiver.into_iter()
                }
            }
        }

        use m2::ThreadIterator;
        vec!["0".to_string(); 10]
            .into_iter()
            .spawn_thread()
            .map(|x| {
                println!("id {:?} receive1 {}", std::thread::current().id(), x);
                x
            })
            .spawn_thread()
            .map(|x| {
                println!("id {:?} receive2 {}", std::thread::current().id(), x);
                x
            })
            .spawn_thread()
            .for_each(|x| {
                println!("id {:?} receive3 {}", std::thread::current().id(), x);
            });

        type MAP = std::collections::HashMap<String, String>;
        fn handle_n_times(
            count: u32,
            map: std::sync::Arc<std::sync::Mutex<MAP>>,
        ) -> std::vec::Vec<std::thread::JoinHandle<()>> {
            let mut handler_vec: std::vec::Vec<std::thread::JoinHandle<()>> = std::vec::Vec::new();
            for i in 0..count {
                let map = map.clone(); // map need to be cloned, since map is moved in thread closure
                handler_vec.push(std::thread::spawn(move || {
                    let mut map: std::sync::MutexGuard<MAP> = map.lock().unwrap();
                    map.insert(format!("key-{}", i), format!("value-{}", i));
                }));
            }
            handler_vec
        }
        let map = std::sync::Arc::new(std::sync::Mutex::new(MAP::new()));
        handle_n_times(5, map.clone()).into_iter().for_each(|x| {
            x.join();
        }); // map need to be cloned, since map is moved in function, cannot be used outside anymore
        dbg!(map);

        {
            use std::sync::mpsc::{channel, Receiver, Sender};
            use std::sync::{Arc, Mutex};

            #[derive(Clone)]
            pub struct SharedReceiver<T>(Arc<Mutex<Receiver<T>>>);

            impl<T> Iterator for SharedReceiver<T> {
                type Item = T;
                fn next(&mut self) -> Option<T> {
                    let guard = self.0.lock().unwrap();
                    guard.recv().ok()
                }
            }

            impl<T> SharedReceiver<T> {
                pub fn shared_channel() -> (Sender<T>, SharedReceiver<T>) {
                    let (sender, receiver) = channel();
                    (sender, SharedReceiver(Arc::new(Mutex::new(receiver))))
                }
            }
        }

        {
            let mutex_condvar =
                std::sync::Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
            let local_mutex_condvar = mutex_condvar.clone();
            std::thread::spawn(move || {
                let mut flag: std::sync::MutexGuard<bool> = mutex_condvar.0.lock().unwrap();
                *flag = true;
                mutex_condvar.1.notify_all();
                println!("set to true and notify");
            });
            let mut flag: std::sync::MutexGuard<bool> = local_mutex_condvar.0.lock().unwrap();
            // while *flag != true {
            //     flag = local_mutex_condvar.1.wait(flag).unwrap();
            // }
            local_mutex_condvar.1.wait_while(flag, |x| *x == false);
            println!("get true");
        }

        {
            let cancel_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let worker_cancel_flag = cancel_flag.clone();
            let worker_handle = std::thread::spawn(move || {
                for i in 0..100 {
                    std::thread::sleep(std::time::Duration::from_secs(3));
                    if worker_cancel_flag.load(std::sync::atomic::Ordering::SeqCst) {
                        println!("stop");
                        return;
                    }
                    println!("continue");
                }
            });
            std::thread::sleep(std::time::Duration::from_secs(8));
            cancel_flag.store(true, std::sync::atomic::Ordering::SeqCst);
            println!("send stop");
            worker_handle.join().unwrap();
            println!("complete");
        }

        println!(
            "start GLOBAL_VAR {}",
            GLOBAL_VAR.load(std::sync::atomic::Ordering::SeqCst)
        );
        let worker_handle = std::thread::spawn(move || {
            GLOBAL_VAR.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let mut var = HOSTNAME.lock().unwrap();
            var.push_str("_inthread");
        });
        worker_handle.join().unwrap();
        println!(
            "running GLOBAL_VAR {}",
            GLOBAL_VAR.load(std::sync::atomic::Ordering::SeqCst)
        );
        let var = HOSTNAME.lock().unwrap();
        println!("HOSTNAME {}", *var);
    }
    return;
    {
        let mut vec = vec!["".to_string(); 100];
        vec.into_iter().for_each(|mut x| {
            x.push('-');
        });
        dbg!(vec);
        let mut vec = vec!["".to_string(); 100];
        vec.iter_mut().for_each(|x| {
            x.push('-');
        });
        dbg!(vec);
    }
    return;
    {
        use std::sync::Arc;
        type ArcStringVec = std::vec::Vec<Arc<String>>;
        fn process_files_parallel(names: &ArcStringVec) {
            const THREADS_NUM: usize = 2;
            let works = split_works(names, THREADS_NUM);
            let mut th_hand_vec = std::vec::Vec::new();
            for (_, names) in works {
                th_hand_vec.push(std::thread::spawn(move || {
                    for name in names {
                        println!("complete {}", name);
                        std::thread::sleep(std::time::Duration::from_secs(1));
                    }
                }));
            }
            for item in th_hand_vec {
                item.join().expect("thread error");
            }
        }
        fn split_works(
            names: &ArcStringVec,
            threads_num: usize,
        ) -> std::collections::HashMap<usize, ArcStringVec> {
            let mut map = std::collections::HashMap::<usize, ArcStringVec>::new();
            for i in 0..threads_num {
                map.insert(i, ArcStringVec::new());
            }
            let mut idx: usize = 0;
            for name in names {
                let vec = map.get_mut(&idx).unwrap();
                vec.push(name.clone());
                idx = (idx + 1) % threads_num;
            }
            map
        }
        let vec: ArcStringVec = vec!["1", "2", "3", "4", "5", "6"]
            .iter()
            .map(|&x| Arc::new(String::from(x)))
            .collect();
        // process_files_parallel(&vec);
        let names = ["1", "2", "3", "4", "5", "6"];
        use rayon::prelude::*;
        names.par_iter().for_each(|&name| {
            println!("complete {}", name);
            std::thread::sleep(std::time::Duration::from_secs(1));
        });
    }
    return;
    {
        const BUF_SIZE: usize = 4 * 1024;
        fn copy<T1, T2>(
            reader: &mut T1,
            writer: &mut T2,
        ) -> Result<usize, Box<dyn std::error::Error>>
        where
            T1: std::io::Read,
            T2: std::io::Write,
        {
            let mut buf: [u8; BUF_SIZE] = [0; BUF_SIZE];
            let mut total_size: usize = 0;
            loop {
                let read_size = match reader.read(&mut buf) {
                    Ok(0) => return Ok(total_size),
                    Ok(size) => size,
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(e) => return Err(Box::new(e)),
                };
                writer.write_all(&buf[..read_size])?;
                total_size += read_size;
            }
        }
        // let mut srcf = std::fs::File::open("test.txt").expect("open test.txt error");
        // let mut destf = std::fs::File::create("test.out").expect("open test.out error");
        // copy(&mut srcf, &mut destf).expect("copy error");

        fn grep<R: std::io::BufRead>(pattern: &str, reader: &mut R) -> std::io::Result<()> {
            use std::io::BufRead;
            for line in reader.lines() {
                let line_content = line?;
                if let Some(_) = line_content.find(pattern) {
                    println!("{}", line_content);
                }
            }
            Ok(())
        }
        // grep("test", &mut std::io::stdin().lock()).unwrap();
        let file = std::fs::File::open("rust.txt").expect("open rust.txt error");
        // grep("h", &mut std::io::BufReader::new(file)).expect("grep error");
        {
            fn grep<R>(target: &str, reader: R) -> std::io::Result<()>
            where
                R: std::io::BufRead,
            {
                for line_result in reader.lines() {
                    let line = line_result?;
                    if line.contains(target) {
                        println!("{}", line);
                    }
                }
                Ok(())
            }
            fn grep_main() -> Result<(), Box<dyn std::error::Error>> {
                // 获取命令行参数。第一个参数是要搜索的字符串；剩余的是文件名。
                let mut args = std::env::args().skip(1);
                let target = match args.next() {
                    Some(s) => s,
                    None => Err("usage: grep PATTERN FILE...")?,
                };
                let files: Vec<std::path::PathBuf> = args.map(std::path::PathBuf::from).collect();
                if files.is_empty() {
                    let stdin = std::io::stdin().lock();
                    grep(&target, stdin)?;
                } else {
                    for file in files {
                        let f = std::fs::File::open(file)?;
                        grep(&target, std::io::BufReader::new(f))?;
                    }
                }
                Ok(())
            }
        }
        fn test_lines_collect(
            reader: &mut dyn std::io::BufRead,
        ) -> Result<(), Box<dyn std::error::Error>> {
            use std::io::BufRead;
            // let lines = reader.lines().collect::<std::io::Result<Vec<String>>>();
            // let lines = reader.lines().collect::<Vec<Result<String, std::io::Error>>>();
            let lines: Vec<String> = reader.lines().filter_map(|x| x.ok()).collect();
            println!("{:#?}", lines);
            Ok(())
        }
        let mut file = std::fs::File::open("rust.txt").unwrap();
        test_lines_collect(&mut std::io::BufReader::new(file));
        fn test_fs() -> std::io::Result<()> {
            // let mut filename = "test.log";
            // let mut file = std::fs::File::open(filename)?;
            // let mut file = std::fs::File::create(filename)?;
            // let mut file = std::fs::OpenOptions::new()
            //     .append(true)
            //     .open("server.log")?;
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open("new_file.txt")?;
            use std::io::Seek;
            let new_pos = file.seek(std::io::SeekFrom::Start(100))?;
            use std::io::Write;
            file.write_all("test string".as_bytes())?;
            drop(file);
            Ok(())
        }
        // test_fs().unwrap();
        fn test_command() -> std::io::Result<()> {
            let mut child = std::process::Command::new("grep")
                .arg("-e")
                .arg("a.*e.*i.*o.*u")
                .stdin(std::process::Stdio::piped())
                .spawn()?;
            let mut to_child = child.stdin.take().unwrap();
            use std::io::Write;
            for word in ["my_words"] {
                to_child.write_fmt(format_args!("{}", word))?;
            }
            drop(to_child); // close grep's stdin, so it will exit
            child.wait()?;
            Ok(())
        }
        {
            fn test_byteorder_flate2<R: std::io::Read, W: std::io::Write>(
                mut reader: R,
                mut writer: W,
            ) -> Result<(), Box<dyn std::error::Error>> {
                use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
                let n = reader.read_u32::<LittleEndian>()?;
                writer.write_i64::<LittleEndian>(n as i64)?;
                use flate2::read::GzDecoder;
                let mut gzip_reader = GzDecoder::new(reader);
                Ok(())
            }
            fn test_serde() {
                let mut map: std::collections::HashMap<String, String> =
                    std::collections::HashMap::new();
                map.insert("1".into(), "1111".into());
                map.insert("2".into(), "2222".into());
                serde_json::to_writer_pretty(std::io::stdout().lock(), &map);
                #[derive(serde::Serialize, serde::Deserialize)]
                struct Player {
                    location: String,
                    items: Vec<String>,
                    health: u32,
                }
            }
            // test_serde();
        }
        {
            fn process_file<P>(path_arg: P) -> std::io::Result<()>
            where
                P: AsRef<std::path::Path>,
            {
                let path = path_arg.as_ref();
                let mut path: std::path::PathBuf = std::path::Path::new("..").into();
                let abs_path = std::env::current_dir()?.join("..");
                let file = std::path::Path::new("/home/jimb/calendars/calendar-18x18.pdf");
                assert_eq!(
                    file.ancestors().collect::<Vec<_>>(),
                    vec![
                        std::path::Path::new("/home/jimb/calendars/calendar-18x18.pdf"),
                        std::path::Path::new("/home/jimb/calendars"),
                        std::path::Path::new("/home/jimb"),
                        std::path::Path::new("/home"),
                        std::path::Path::new("/")
                    ]
                );
                for entry_result in path.read_dir()? {
                    let entry = entry_result?;
                    println!("{}", entry.file_name().to_string_lossy());
                }
                Ok(())
            }
            fn test_copy_dir() -> std::io::Result<()> {
                fn copy_dir_to<P: AsRef<std::path::Path>>(src: P, dst: P) -> std::io::Result<()> {
                    let src = src.as_ref();
                    let dst = dst.as_ref();
                    if !dst.is_dir() {
                        std::fs::create_dir(dst)?;
                    }
                    for entry_result in src.read_dir()? {
                        let entry = entry_result?;
                        let file_type = entry.file_type()?;
                        copy_to(&entry.path(), &file_type, &dst.join(entry.file_name()))?;
                    }
                    return Ok(());
                    fn copy_to(
                        src: &std::path::Path,
                        src_type: &std::fs::FileType,
                        dst: &std::path::Path,
                    ) -> std::io::Result<()> {
                        if src_type.is_file() {
                            std::fs::copy(src, dst)?;
                        } else if src_type.is_dir() {
                            copy_dir_to(src, dst)?;
                        } else {
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::Other,
                                format!("don't know how to copy: {}", src.display()),
                            ));
                        }
                        Ok(())
                    }
                }
                copy_dir_to("src", "copyto");
                Ok(())
            }
            // test_copy_dir().unwrap();
        }
        {
            fn test_tcp_echo_server(addr: &str) -> std::io::Result<()> {
                let listener = std::net::TcpListener::bind(addr)?;
                println!("listening on {}", addr);
                loop {
                    let (mut stream, addr) = listener.accept()?;
                    println!("connection received from {}", addr);
                    let mut write_stream = stream.try_clone()?;
                    std::thread::spawn(move || {
                        std::io::copy(&mut stream, &mut write_stream)
                            .expect("error in client thread: ");
                        println!("connection closed");
                        drop(stream);
                        drop(write_stream);
                    });
                }
                Ok(())
            }
            fn test_tcp_echo_client(addr: &str) -> std::io::Result<()> {
                let mut stream = std::net::TcpStream::connect(addr)?;
                println!("connect to {}", addr);
                use std::io::Write;
                stream.write_all("Rust TCP Client!".as_bytes())?;
                let mut buf: [u8; 4096] = [0; 4096];
                use std::io::Read;
                let rsize = stream.read(&mut buf)?;
                if rsize > 0 {
                    let str = std::str::from_utf8(&buf[..rsize]).expect("convert response error");
                    print!("receive {}", str);
                }
                drop(stream);
                Ok(())
            }
            match std::env::args().skip(1).next() {
                Some(type_) if type_ == "server" => {
                    test_tcp_echo_server("0.0.0.0:8088").unwrap();
                }
                Some(type_) if type_ == "client" => {
                    test_tcp_echo_client("0.0.0.0:8088").unwrap();
                    // let mut response = reqwest::blocking::get("http://127.0.0.1:8088/").expect("http get error");
                    // if !response.status().is_success() {
                    //     eprintln!("{}", response.status());
                    // } else {
                    //     let stdout = std::io::stdout();
                    //     std::io::copy(&mut response, &mut stdout.lock()).expect("copy response error");
                    // }
                }
                _ => (),
            };
        }
    }
    return;
    {
        #[derive(Debug)]
        struct MyError(String);
        impl std::error::Error for MyError {
            fn description(&self) -> &str {
                self.0.as_str()
            }
        }
        impl std::fmt::Display for MyError {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", &self.0)
            }
        }
        let e: Result<(), MyError> = Err(MyError("panic".to_string()));
        e.unwrap();
    }
    return;
    {
        fn test_regex() -> Result<(), Box<dyn std::error::Error>> {
            let ver_regex = regex::Regex::new(r#"(\d+)\.(\d+)\.(\d+)(-[-.[:alnum:]]*)?"#)?;
            let ver_str = r#"regex = "0.2.5""#;
            println!("match result: {}", ver_regex.is_match(ver_str));
            let captures = ver_regex.captures(ver_str).ok_or("captures error")?;
            captures.iter().for_each(|x| {
                x.map_or((), |v| {
                    println!("{} start {} end {}", v.as_str(), v.start(), v.end())
                })
            });

            let ver_str = "In the beginning, there was 1.0.0. \
 For a while, we used 1.0.1-beta, \
 but in the end, we settled on 1.2.4.";
            let matches: Vec<&str> = ver_regex
                .find_iter(ver_str)
                .map(|match_| match_.as_str())
                .collect();
            println!("{matches:?}");

            let re = regex::Regex::new(r"'(?P<title>[^']+)'\s+\((?P<year>\d{4})\)").unwrap();
            let text = "'Citizen Kane' (1941), 'The Wizard of Oz' (1939), 'M' (1931)., 'test'";
            for caps in re.captures_iter(text) {
                println!("Movie: {:?}, Released: {:?}", &caps["title"], &caps["year"]);
            }

            Ok(())
        }
        test_regex().expect("test_regex error");
    }
    return;
    {
        println!("[{:2}]", "bookend"); // [bookend]
        println!("[{:12}]", "bookend"); // [bookend     ]
        println!("[{:_<12}]", "bookend"); // [bookend_____]
        println!("[{:*>12}]", "bookend"); // [*****bookend]
        println!("[{:4.6}]", "bookend"); // [booken]
        println!("[{:$^6.4}]", "bookend"); // [$book$]
        println!("[{:_^6.3}]", "测试数据"); // [_测试数__]

        println!("[{:2}]", 1234); // [1234]
        println!("[{:6}]", 1234); // [  1234]
        println!("[{:+06}]", 1234); // [+01234]
        println!("[{:<6}]", 1234); // [1234  ]
        println!("[{:_^+6}]", 1234); // [+1234_]
        println!("[{:b}]", 1234); // [10011010010]
        println!("[{:#o}]", 1234); // [0o2322]
        println!("[{:#x}]", 1234); // [0x4d2]
        println!("[{:06X}]", 1234); // [0004D2]

        println!("[{:.2}]", 1234.5678); // [1234.57]
        println!("[{:.6}]", 1234.5678); // [1234.567800]
        println!("[{:+12}]", 1234.5678); // [  +1234.5678]
        println!("[{:012}]", 1234.5678); // [0001234.5678]
        println!("[{:e}]", 1234.5678); // [1.2345678e3]
        println!("[{:.2e}]", 1234.5678); // [1.23e3]
        println!("[{:012.2e}]", 1234.5678); // [0000001.23e3]

        println!("[{:?}]", [(10, 11), (20, 22), (30, 33)]); // [[(10, 11), (20, 22), (30, 33)]]
        println!("[{:#x?}]", [(10, 11)]); // pretty-print [[\n(\n0xa,\n0xb,\n),\n]]
        println!("[{:+06X?}]", [(10, 11), (20, 22), (30, 33)]); // [[(+0000A, +0000B), (+00014, +00016), (+0001E, +00021)]]

        let original = std::rc::Rc::new("mazurka".to_string());
        let cloned = original.clone();
        let impostor = std::rc::Rc::new("mazurka".to_string());
        println!("pointers: {:p}, {:p}, {:p}", original, cloned, impostor); // pointers: 0x55a4ff905af0, 0x55a4ff905af0, 0x55a4ff905a20

        struct MyStruct {
            val: String,
        }
        let mut ptr: Box<Vec<String>> = Box::new(std::vec::Vec::new());
        ptr.push("1".to_string());
        dbg!(&ptr);
        ptr.push("2".to_string());
        dbg!(&ptr);
        let mut ptr: std::rc::Rc<Vec<String>> = std::rc::Rc::new(std::vec::Vec::new());
        dbg!(&ptr);

        assert_eq!(
            format!("{2:#06x},{1:b},{0:=>10}", "first", 10, 100),
            "0x0064,1010,=====first"
        );
        assert_eq!(
            format!(
                "{mode} {2} {} {}",
                "people",
                "eater",
                "purple",
                mode = "flying"
            ),
            "flying purple people eater"
        );
        println!("[{:<1$}]", "bookend", 8); // [bookend ]
        println!("[{:^width$.limit$}]", "bookend", width = 8, limit = 3); // [  boo   ]
        println!("[{:>.*}]", 4, "bookend"); // [book]
        println!("[{:^8.*}]", 2, 1234.5678); // [1234.57 ]
        {
            struct Complex {
                re: f64,
                im: f64,
            }
            impl std::fmt::Display for Complex {
                fn fmt(&self, dest: &mut std::fmt::Formatter) -> std::fmt::Result {
                    let (re, im) = (self.re, self.im);
                    if dest.alternate() {
                        let abs = f64::sqrt(re * re + im * im);
                        let angle = f64::atan2(im, re) / std::f64::consts::PI * 180.0;
                        write!(dest, "{} ∠ {}°", abs, angle)
                    } else {
                        let im_sign = if im < 0.0 { '-' } else { '+' };
                        write!(dest, "{} {} {}i", re, im_sign, f64::abs(im))
                    }
                }
            }
            let ninety = Complex { re: 0.0, im: 2.0 };
            println!("{}", ninety);
            println!("{:#}", ninety);
        }
        {
            use std::io::Write;
            fn write_log_entry(entry: std::fmt::Arguments) -> bool {
                if let Ok(mut log_file) = std::fs::OpenOptions::new()
                    .append(true)
                    .create(true)
                    .open("test.log")
                {
                    if log_file.write_fmt(entry).is_ok() {
                        return true;
                    }
                }
                false
            }
            write_log_entry(format_args!("test log {:?}", vec![1, 2, 3, 4]));
        }
    }
    return;
    {
        assert_eq!(
            "うどん: udon".as_bytes(),
            &[
                0xe3, 0x81, 0x86, // う
                0xe3, 0x81, 0xa9, // ど
                0xe3, 0x82, 0x93, // ん
                0x3a, 0x20, 0x75, 0x64, 0x6f, 0x6e // : udon
            ]
        );
        assert_eq!("カニ".chars().next(), Some('カ'));
        let str = "测试";
        str.bytes().for_each(|x| print!("{:X} ", x));
        println!("");
        let str = "\u{E6}\u{B5}\u{8B}\u{E8}\u{AF}\u{95}";
        dbg!(str.bytes());
        dbg!(str);

        assert_eq!('F'.to_digit(16), Some(15));
        assert_eq!(std::char::from_digit(15, 16), Some('f'));
        assert!(char::is_digit('f', 16));

        let mut upper = 's'.to_uppercase();
        assert_eq!(upper.next(), Some('S'));
        assert_eq!(upper.next(), None);

        assert_eq!(char::from(66), 'B');
        assert_eq!(std::char::from_u32(0x9942), Some('饂'));
        assert_eq!(std::char::from_u32(0xd800), None); // reserved for UTF-16

        if let Some(val) = 'a'.to_digit(16) {
            println!("to_digit {val}");
        }
        if let Some(val) = std::char::from_digit(10, 16) {
            println!("from_digit {val}");
        }

        let space_sentence = "man hat tan";
        let spaceless_sentence: String = space_sentence
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();
        assert_eq!(spaceless_sentence, "manhattan");

        let full = "bookkeeping";
        assert_eq!(&full[..4], "book");
        assert_eq!(&full[5..], "eeping");
        assert_eq!(&full[2..4], "ok");
        assert_eq!(full[..].len(), 11);
        assert_eq!(full[5..].contains("boo"), false);

        let str = "测试句子";
        // println!("{}", &str[1..]); // panic
        println!("{}", &str[3..]); // 试句子
        let sub_str: String = str.chars().skip(2).collect();
        println!("{}", &sub_str); // 句子
        let str = "测试句子".to_string();
        println!("{}", &str[3..]); // 试句子

        let mut str = "con".to_string();
        str.extend("tri but ion".split(" "));
        dbg!(&str);

        use std::fmt::Write;
        let mut letter = String::new();
        writeln!(letter, "Whose {} these are I think I know", "rutabagas")
            .expect("write to String error");
        writeln!(letter, "His house is in the village though;").expect("write to String error");
        assert_eq!(
            letter,
            "Whose rutabagas these are I think I know\n\
             His house is in the village though;\n"
        );
        let left = "partners".to_string();
        let mut right = "crime".to_string();
        assert_eq!(left + " in " + &right, "partners in crime");
        right += " doesn't pay";
        assert_eq!(right, "crime doesn't pay");

        // let parenthetical = "(" + str + ")"; // error
        let parenthetical = "(".to_string() + &str + ")"; // ok

        let mut str = "测试句子".to_string();
        // str.truncate(4); // panic, is_char_boundary check failed
        let str: String = str.chars().take(2).collect();
        println!("{}", &str); // 测
        let mut str = "123".to_string();
        let char = str.remove(1);
        dbg!(&char);
        dbg!(&str);

        let mut choco = "chocolate".to_string();
        choco.drain(3..4);
        dbg!(&choco);
        assert_eq!(choco.drain(3..5).collect::<String>(), "ol");
        assert_eq!(choco, "choate");

        let haystack = "One fine day, in the middle of the night";
        assert_eq!(haystack.find(','), Some(12));
        assert_eq!(haystack.find("night"), Some(35));
        assert_eq!(haystack.find(char::is_whitespace), Some(3));
        assert_eq!(haystack.find(['y', 'e']), Some(2));
        let code = "\t function noodle() { ";
        assert_eq!(code.trim_start_matches([' ', '\t']), "function noodle() { ");
        assert!("2017".starts_with(char::is_numeric));
        let quip = "We also know there are known unknowns";
        assert_eq!(quip.find("know"), Some(8));
        assert_eq!(quip.rfind("know"), Some(31));
        assert_eq!(quip.find("ya know"), None);
        assert_eq!(quip.rfind(char::is_uppercase), Some(0));
        assert_eq!(
            "The only thing we have to fear is fear itself".replace("fear", "spin"),
            "The only thing we have to spin is spin itself"
        );
        assert_eq!(
            "`Borrow` and `BorrowMut`".replace(|ch: char| !ch.is_alphanumeric(), ""),
            "BorrowandBorrowMut"
        );
        assert_eq!(
            "测试".char_indices().collect::<Vec<_>>(),
            vec![(0, '测'), (3, '试')]
        );
        "_1_2_".split('_').for_each(|x| print!("[{}]", x)); // [][1][2][]
        println!("");
        "_1_2_"
            .split_terminator('_')
            .for_each(|x| print!("[{}]", x)); // [][1][2]
        println!("");
        "_1_2_".splitn(2, '_').for_each(|x| print!("[{}]", x)); // [][1_2_]
        println!("");
        "  1  2  "
            .split_whitespace()
            .for_each(|x| print!("[{}]", x)); // [1][2]
        println!("");
        "_1_2_".matches("_").for_each(|x| print!("[{}]", x)); // [_][_][_]
        println!("");
        "_1_2_"
            .rmatches(|x: char| x.is_digit(10))
            .for_each(|x| print!("[{}]", x)); // [2][1]
        println!("");
        println!("count {}", "_1_2_".matches("_1").count());
        if let Some((idx, _)) = "_1_2_".match_indices("_1").last() {
            println!("count {}", idx + 1);
        }

        let str_vec = "1 2 3 4 5 6".split_whitespace().collect::<Vec<_>>();

        assert_eq!("\t*.rs ".trim(), "*.rs");
        assert_eq!("\t*.rs ".trim_start(), "*.rs ");
        assert_eq!("\t*.rs ".trim_end(), "\t*.rs");
        assert_eq!("0120 34 ".trim_matches(&['0', ' '][..]), "120 34");
        "01 203 4"
            .matches(['0', ' '])
            .for_each(|x| print!("[{}]", x));
        println!("");
        let str = "01 203 4".trim_matches(['0', ' '].as_ref());
        dbg!(&str);

        {
            use std::str::FromStr;
            assert_eq!(usize::from_str("3628800"), Ok(3628800));
            assert_eq!(f64::from_str("128.5625"), Ok(128.5625));
            assert_eq!(bool::from_str("true"), Ok(true));
            assert!(f64::from_str("not a float at all").is_err());
            assert!(bool::from_str("TRUE").is_err());
            assert_eq!(char::from_str("é"), Ok('é'));
            assert!(char::from_str("abcdefg").is_err());
        }
        {
            let val = "AA".parse::<i32>().unwrap_or(-1);
        }
        let str = format!("({:.3}, {:.3})", 0.5, f64::sqrt(3.0) / 2.0);
        let vec = vec![1, 2, 3];
        let str = format!("{:?}", &vec);
        #[derive(Debug)]
        struct MyStruct {
            val: String,
        }
        impl std::fmt::Display for MyStruct {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.val)
            }
        }
        let str = MyStruct {
            val: "test".to_string(),
        }
        .to_string();
        dbg!(&str);

        let str = "测试数据".to_string();
        let mut buf = str.into_bytes();
        // dbg!(&str); // error, str inner data is moved

        fn get_env(name: &str) -> std::borrow::Cow<'static, str> {
            std::env::var(name)
                .map(|x| std::borrow::Cow::Owned(x))
                .unwrap_or(std::borrow::Cow::Borrowed("get_env failed"))
        }
        let mut env_str = get_env("test");
        println!("result: {}", env_str);
        env_str.to_mut().push('\n');
        env_str += "another line";
        println!("new result: {}", env_str);
        let val: Vec<_> = ["1", "2"].iter().map(|&x| x.to_owned()).collect();
        let val_str: std::borrow::Cow<String> =
            std::borrow::Cow::Owned(format!("value is {}", 1.00f64));
        let mut val_str_vec: Vec<std::borrow::Cow<String>> = Vec::new();
        val_str_vec.push(val_str);
    }
    return;
    {
        let mut old_heap: std::collections::BinaryHeap<i32> = std::collections::BinaryHeap::new();
        old_heap.extend([2, 3, 8].iter());
        [21, 46, 63].iter().for_each(|&x| old_heap.push(x));
        let mut heap = std::collections::BinaryHeap::from(vec![6, 9, 5, 4]);
        heap.append(&mut old_heap);
        if let Some(mut top) = heap.peek_mut() {
            *top += 10;
            std::collections::binary_heap::PeekMut::pop(top);
        }
        println!("top: {}", heap.peek().unwrap_or(&-1));

        let vec = [("1".to_string(), 1), ("2".to_string(), 2)];
        let mut map = std::collections::HashMap::from(vec);
        dbg!(&map);
        let entry = map.entry("1".into());
        entry.or_insert(10);
        dbg!(&map);
        let entry = map.entry("1".into());
        let val_ref = entry.or_insert(10);
        *val_ref += 10;
        dbg!(&map);

        let mut word_frequency =
            std::collections::HashMap::from([("a".to_string(), 0), ("b".to_string(), 0)]);
        let text = "a\na b\nb\nb\nc c c";
        for line in text.lines() {
            let entry = word_frequency.entry(line.to_string());
            let val_ref = entry.and_modify(|v| *v += 1).or_insert(0);
            if *val_ref == 0 {
                *val_ref = 1;
            }
        }
        dbg!(&word_frequency);
        word_frequency
            .iter()
            .for_each(|(k, v)| println!("{}-{}", k, v));
        let mut btree_map: std::collections::BTreeMap<&String, &i32> =
            std::collections::BTreeMap::new();
        word_frequency.iter().for_each(|(k, v)| {
            btree_map.insert(k, v);
        });
        btree_map
            .iter()
            .for_each(|(&k, &v)| println!("{}-{}", k, v));

        #[derive(Clone, PartialEq, Eq, Hash)]
        struct MyStruct {
            val1: String,
        }
        type MuseumNumber = u32;
        type Culture = String;
        struct Artifact {
            id: MuseumNumber,
            name: String,
            cultures: Vec<Culture>,
        }
        impl PartialEq for Artifact {
            fn eq(&self, other: &Artifact) -> bool {
                self.id == other.id
            }
        }
        impl Eq for Artifact {}
        impl std::hash::Hash for Artifact {
            fn hash<H: std::hash::Hasher>(&self, hasher: &mut H) {
                self.id.hash(hasher); // 把哈希操作委托给 MuseumNumber
            }
        }
    }
    return;
    {
        let v = std::collections::VecDeque::from(vec![1, 2, 3, 4]);
        let v = std::collections::VecDeque::from([1, 2, 3, 4]);

        use std::time::{SystemTime, UNIX_EPOCH};
        let time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        println!("timestamp: {time}");
        use chrono::prelude::*;
        let time = Utc::now().timestamp();
        println!("timestamp: {time}");
    }
    return;
    {
        // Vec
        #[derive(Debug, Clone)]
        enum Spec {
            Int(i32),
            Float(f64),
            Bool(bool),
            Text(String),
        }
        impl std::fmt::Display for Spec {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match *self {
                    Spec::Int(val) => write!(f, "Int {}", val),
                    Spec::Float(val) => write!(f, "Float {}", val),
                    Spec::Bool(val) => write!(f, "Bool {}", val),
                    Spec::Text(ref val) => write!(f, "Text {}", val),
                }
            }
        }
        let mut v = vec![
            Spec::Int(1),
            Spec::Float(2.2),
            Spec::Bool(true),
            Spec::Text(String::from("hello")),
        ];
        println!("{:?}", &v);
        dbg!(&v);
        let first = v[3].clone();
        v.push(Spec::Int(6));
        println!("The first element is: {}", first);

        let mut my_vec = vec![1, 3, 5, 7, 9];
        for (index, &val) in my_vec.iter().enumerate() {
            if val > 4 {
                // my_vec.remove(index); // error: can't borrow `my_vec` as mutable
            }
        }
        println!("{:?}", my_vec);

        let v = vec!["1", "2", "3"];
        let v_slice = &v[..1];
        let v_copy = v[..1].to_vec();
        if let Some(item) = v.first() {
            println!("We got one! {}", item);
        }
        let slice = [0, 1, 2, 3];
        assert_eq!(slice.get(2), Some(&2));
        assert_eq!(slice.get(4), None);
        let mut slice = [0, 1, 2, 3];
        {
            let last = slice.last_mut().unwrap(); // 最后一个元素类型：&mut i32
            assert_eq!(*last, 3);
            *last = 100;
        }
        let v = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        assert_eq!(v.to_vec(), vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(v[0..6].to_vec(), vec![1, 2, 3, 4, 5, 6]);
        let mut byte_vec = b"Missssssissippi".to_vec();
        byte_vec.dedup();
        assert_eq!(&byte_vec, b"Misisipi");
        let mut byte_vec = b"Missssssissippi".to_vec();
        let mut seen = std::collections::HashSet::new();
        byte_vec.retain(|r| seen.insert(*r));
        assert_eq!(&byte_vec, b"Misp");
        assert_eq!([[1, 2], [3, 4], [5, 6]].concat(), vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(
            [[1, 2], [3, 4], [5, 6]].join(&0),
            vec![1, 2, 0, 3, 4, 0, 5, 6]
        );
        let v = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
        let sep = &vec![9, 9][..];
        assert_eq!(
            [[1, 2], [3, 4], [5, 6]].join(sep),
            vec![1, 2, 9, 9, 3, 4, 9, 9, 5, 6]
        );
        let v = v.join(sep);
        dbg!(v);
        let v = vec!["1", "2"];
        let v = v.join("--");
        dbg!(v);
        // 很容易一次获得数组、切片、vector中的很多元素的非 mut 引用
        let v = vec![0, 1, 2, 3];
        let (i, j) = (1, 3);
        let a = &v[i];
        let b = &v[j];
        let mid = v.len() / 2;
        let front_half = &v[..mid];
        let back_half = &v[mid..];
        // 但一次获得多个 mut 引用不是这么容易：
        // let mut v = vec![0, 1, 2, 3];
        // let a = &mut v[i];
        // let b = &mut v[j]; // error: 不能同时借用`v`的多个可变引用。
        // *a = 6; // 引用`a`和`b`在这里使用了，
        // *b = 7; // 因此它们的生命周期一定会重叠。
        let daily_high_temperatures = [30, 31, 32, 33];
        let changes = daily_high_temperatures
            .windows(2) // 获得相邻天的温度
            .map(|w| w[1] - w[0]) // 温度改变了多少
            .collect::<Vec<_>>();
        dbg!(changes);

        // 切片
        let v/* : [i32; 4] */ = [1, 2, 3, 4]; // v is array [i32;4]
        let v/* : &[i32; 4] */ = &[1, 2, 3, 4]; // v is ref of array [i32;4]
        let v/* : &[i32] */ = &[1, 2, 3, 4][..]; // v is slice &[i32]
        let v/* : &[i32] */ = [1, 2, 3, 4].as_ref(); // v is slice &[i32]
        let v_slice: &[i32] = &[1, 2, 3, 4];
        // v_slice.sort(); // `v` is a `&` reference, so the data it refers to cannot be borrowed as mutable
        let v_slice: &mut [i32] = &mut [1, 2, 3, 4];
        v_slice.sort();
        let v/* : &[u8; 4] */ = b"1234"; // v is ref of array [u8; 4]
        let v_slice: &[u8] = b"1234";

        #[derive(Debug)]
        struct Student {
            first_name: String,
            last_name: String,
        }
        let mut students = [
            Student {
                first_name: "f".to_string(),
                last_name: "c".to_string(),
            },
            Student {
                first_name: "e".to_string(),
                last_name: "c".to_string(),
            },
        ];
        dbg!(&students);
        students.sort_by(|a, b| {
            let a_key = (&a.last_name, &a.first_name);
            let b_key = (&b.last_name, &b.first_name);
            a_key.cmp(&b_key)
        });
        dbg!(&students);
        let mut v = &[1, 2, 3, 4][..];
        // v.sort(); // `v` is a `&` reference, so the data it refers to cannot be borrowed as mutable
        v.binary_search_by_key(&4, |x| *x);
        assert_eq!([1, 2, 3, 4].starts_with(&[1, 2]), true);
        assert_eq!([1, 2, 3, 4].ends_with(&[3, 4]), true);
        println!("rand {}", rand::random::<u16>());
        {
            use rand::Rng;
            let val = rand::thread_rng().gen::<i32>();
            dbg!(val);
            let val = rand::thread_rng().gen_range(0.1..99.9);
            dbg!(val);
        }
        {
            use rand::seq::SliceRandom;
            let mut v = [1, 2, 3, 4];
            v.shuffle(&mut rand::thread_rng());
            dbg!(v);
        }
    }
    return;
    {
        struct I32Range {
            start: i32,
            end: i32,
        }
        impl Iterator for I32Range {
            // impl Iterator will also impl IntoIterator
            type Item = i32;
            fn next(&mut self) -> Option<Self::Item> {
                if self.start >= self.end {
                    return None;
                }
                self.start += 1;
                Some(self.start - 1)
            }
        }
        let mut pi: f64 = 0.0;
        let mut numerator: f64 = 1.0;
        for k in (I32Range { start: 0, end: 64 }.into_iter()) {
            pi += numerator / (2 * k + 1) as f64;
            numerator /= -3.0;
        }
        pi *= f64::sqrt(12.0);
        dbg!(pi);

        // An ordered collection of `T`s.
        enum BinaryTree<T> {
            Empty,
            NonEmpty(Box<TreeNode<T>>),
        }
        // A part of a BinaryTree.
        struct TreeNode<T> {
            element: T,
            left: BinaryTree<T>,
            right: BinaryTree<T>,
        }
        impl<T: Ord> BinaryTree<T> {
            fn add(&mut self, value: T) {
                match *self {
                    BinaryTree::Empty => {
                        *self = BinaryTree::NonEmpty(Box::new(TreeNode {
                            element: value,
                            left: BinaryTree::Empty,
                            right: BinaryTree::Empty,
                        }))
                    }
                    BinaryTree::NonEmpty(ref mut node) => {
                        if value <= node.element {
                            node.left.add(value);
                        } else {
                            node.right.add(value);
                        }
                    }
                }
            }
        }
        use BinaryTree::*;
        // The state of an in-order traversal of a `BinaryTree`.
        struct TreeIter<'a, T> {
            // A stack of references to tree nodes. Since we use `Vec`'s
            // `push` and `pop` methods, the top of the stack is the end of the
            // vector.
            //
            // The node the iterator will visit next is at the top of the stack,
            // with those ancestors still unvisited below it. If the stack is empty,
            // the iteration is over.
            unvisited: Vec<&'a TreeNode<T>>,
        }
        impl<'a, T: 'a> TreeIter<'a, T> {
            fn push_left_edge(&mut self, mut tree: &'a BinaryTree<T>) {
                while let NonEmpty(ref node) = *tree {
                    self.unvisited.push(node);
                    tree = &node.left;
                }
            }
        }
        impl<T> BinaryTree<T> {
            fn iter(&self) -> TreeIter<T> {
                let mut iter = TreeIter {
                    unvisited: Vec::new(),
                };
                iter.push_left_edge(self);
                iter
            }
        }
        impl<'a, T> Iterator for TreeIter<'a, T> {
            type Item = &'a T;
            fn next(&mut self) -> Option<&'a T> {
                // Find the node this iteration must produce,
                // or finish the iteration. (Use the `?` operator
                // to return immediately if it's `None`.)
                let node = self.unvisited.pop()?;

                // After `node`, the next thing we produce must be the leftmost
                // child in `node`'s right subtree, so push the path from here
                // down. Our helper method turns out to be just what we need.
                self.push_left_edge(&node.right);

                // Produce a reference to this node's value.
                Some(&node.element)
            }
        }
        impl<'a, T: 'a> IntoIterator for &'a BinaryTree<T> {
            type Item = &'a T;
            type IntoIter = TreeIter<'a, T>;
            fn into_iter(self) -> Self::IntoIter {
                self.iter()
            }
        }
        let mut tree = BinaryTree::Empty;
        tree.add("jaeger");
        tree.add("robot");
        tree.add("droid");
        tree.add("mecha");
        let mut iterator = tree.iter();
        assert_eq!(iterator.next(), Some(&"droid"));
        assert_eq!(iterator.next(), Some(&"jaeger"));
        assert_eq!(iterator.next(), Some(&"mecha"));
        assert_eq!(iterator.next(), Some(&"robot"));
        assert_eq!(iterator.next(), None);
    }
    return;
    {
        use core::fmt::{Display, Formatter, Result};
        #[derive(Debug)]
        enum Platform {
            Linux,
            MacOS,
            Windows,
            Unknown,
        }
        impl Display for Platform {
            fn fmt(&self, f: &mut Formatter) -> Result {
                match self {
                    Platform::Linux => write!(f, "Linux"),
                    Platform::MacOS => write!(f, "MacOS"),
                    Platform::Windows => write!(f, "Windows"),
                    Platform::Unknown => write!(f, "unknown"),
                }
            }
        }
        impl From<Platform> for String {
            fn from(platform: Platform) -> Self {
                match platform {
                    Platform::Linux => "Linux".into(),
                    Platform::MacOS => "MacOS".into(),
                    Platform::Windows => "Windows".into(),
                    Platform::Unknown => "unknown".into(),
                }
            }
        }
        impl From<String> for Platform {
            fn from(platform: String) -> Self {
                match platform {
                    platform if platform == "Linux" => Platform::Linux,
                    platform if platform == "MacOS" => Platform::MacOS,
                    platform if platform == "Windows" => Platform::Windows,
                    platform if platform == "unknown" => Platform::Unknown,
                    _ => Platform::Unknown,
                }
            }
        }
        fn test() {
            let platform: String = Platform::MacOS.to_string();
            println!("{}", platform);
            let platform: String = Platform::Linux.into();
            println!("{}", platform);
            let platform: Platform = "Windows".to_string().into();
            println!("{}", platform);
        }
        test();
    }
    return;
    {
        // use std::io::prelude::BufRead;
        // let stdin = std::io::stdin();
        // println!("{}", stdin.lock().lines().count());
        let text = "1\n2\n3\n4";
        println!("count: {}", text.lines().count());
        println!(
            "sum: {}",
            text.lines()
                .map(|x| x.trim().parse::<i32>().unwrap_or(0))
                .sum::<i32>()
        );
        println!(
            "product: {}",
            text.lines()
                .map(|x| x.trim().parse::<i32>().unwrap_or(1))
                .product::<i32>()
        );

        use std::cmp::Ordering;
        fn cmp(lhs: &f64, rhs: &f64) -> Ordering {
            lhs.partial_cmp(rhs).unwrap() // 比较两个 f64值，如果有 NaN就 panic
        }
        let numbers = [1.0, 4.0, 2.0];
        assert_eq!(numbers.iter().copied().max_by(cmp), Some(4.0));
        assert_eq!(numbers.iter().copied().min_by(cmp), Some(1.0));

        let mut map = std::collections::HashMap::new();
        map.insert("1", 100);
        map.insert("2", 200);
        assert_eq!(map.iter().max_by_key(|x| x.1), Some((&"2", &200)));

        let packed = "Helen of Troy";
        let spaced = "Helen    of  Troy";
        assert!(packed != spaced);
        assert!(packed.split_whitespace().eq(spaced.split_whitespace()));

        let text = "Xerxes";
        assert_eq!(text.chars().position(|c| c == 'e'), Some(1));
        assert_eq!(text.chars().position(|c| c == 'z'), None);

        let a = [5, 6, 7, 8, 9, 10];
        assert_eq!(a.iter().fold(0, |n, _| n + 1), 6); // count
        assert_eq!(a.iter().fold(0, |n, i| n + i), 45); // sum
        assert_eq!(a.iter().fold(1, |n, i| n * i), 151200); // product
        let v = [
            "Pack", "my", "box", "with", "five", "dozen", "liquor", "jugs",
        ];
        let val1 = v.iter().fold(String::new(), |val, elem| val + elem + "_"); // trailing _
        let val2 = v.join("_"); // no trailing _
        dbg!(val1);
        dbg!(val2);

        // {
        //     let stdin = std::io::stdin();
        //     use std::io::prelude::BufRead;
        //     let sum = stdin
        //         .lock()
        //         .lines()
        //         .try_fold(0, |val, elem| -> Result<i32, Box<dyn std::error::Error>> {
        //             Ok(val + elem.unwrap().trim().parse::<i32>().unwrap())
        //         })
        //         .unwrap();
        //     dbg!(sum);
        // }

        let mut squares = (0..10).map(|i| i * i);
        let val: Vec<_> = squares.clone().collect();
        dbg!(val);
        assert_eq!(squares.nth(4), Some(16));
        assert_eq!(squares.nth(0), Some(25));
        assert_eq!(squares.nth(6), None);

        let val = (1..100).into_iter().find(|x| x % 13 == 0);
        assert_eq!(val, Some(13));
        let val = (1..100).into_iter().find_map(|x| {
            if x % 13 == 0 {
                return Some("Found");
            }
            None
        });
        assert_eq!(val, Some("Found"));

        {
            use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, LinkedList};
            let args: HashSet<String> = std::env::args().collect();
            let args: BTreeSet<String> = std::env::args().collect();
            let args: LinkedList<String> = std::env::args().collect();
            let args: HashMap<String, usize> = std::env::args().zip(0..).collect();
            let args: BTreeMap<String, usize> = std::env::args().zip(0..).collect();
            let args: Vec<(String, usize)> = std::env::args().zip(0..).collect();
            let args: Vec<(usize, String)> = (0..).zip(std::env::args()).collect();
        }
        let mut v: Vec<i32> = (0..5).map(|i| 1 << i).collect();
        v.extend(&[31, 57, 99, 163]);
        assert_eq!(v, &[1, 2, 4, 8, 16, 31, 57, 99, 163]);

        let v = [1, 2, 3, 11, 12, 13];
        let (v1, v2): (Vec<i32>, Vec<i32>) = v.iter().partition(|&x| {
            if *x < 10 {
                return true;
            }
            false
        });
        dbg!(v1);
        dbg!(v2);

        ["doves", "hens", "birds"]
            .iter()
            .zip(["turtle", "french", "calling"].iter())
            .zip(2..5)
            .rev()
            .map(|((item, kind), quantity)| format!("{} {} {}", quantity, kind, item))
            .for_each(|gift| {
                println!("You have received: {}", gift);
            });
        use std::io::Write;
        let mut output: std::fs::File =
            std::fs::File::create("test.txt").expect("create file error");
        ["1", "2", "3", "4"]
            .iter()
            .try_for_each(|&x| write!(&mut output, "{}", x))
            .unwrap();
    }
    return;
    {
        let lines = "1\n2".lines();
        let mut it = lines.into_iter();
        it.by_ref().inspect(|&e| println!("{}", e)).all(|_| true);
        it.by_ref().inspect(|&e| println!("{}", e)).all(|_| true); // empty, since it is consumed totally
        let lines: Vec<_> = "1\n2".lines().into_iter().collect();
        lines.iter().all(|&l| {
            println!("{}", l);
            true
        });
        lines.iter().all(|&l| {
            println!("{}", l);
            true
        });

        //当字符串行以反斜杠结尾时，Rust并不会把下一行的缩进包含进字符串里
        let message = "\
To: jimb\r\n\
From: superego <editor@oreilly.com>\r\n\
\r\n\
Did you get any writing done today?\r\n\
When will you stop wasting time plotting fractals?\r\n";
        for header in message.lines().take_while(|l| !l.is_empty()) {
            println!("header: {}", header);
        }
        for body in message.lines().skip_while(|l| !l.is_empty()).skip(1) {
            println!("body: {}", body);
        }
        let val = std::env::args().skip(1);
        println!("{:#?}", val);

        use std::iter::Peekable;
        fn parse_number<I>(tokens: &mut Peekable<I>) -> u32
        where
            I: Iterator<Item = char>,
        {
            let mut n = 0;
            loop {
                match tokens.peek() {
                    Some(r) if r.is_digit(10) => {
                        n = n * 10 + r.to_digit(10).unwrap();
                    }
                    _ => {
                        return n;
                    }
                }
                tokens.next();
            }
        }
        let mut chars = "226153980,1766319049".chars().peekable();
        assert_eq!(parse_number(&mut chars), 226153980);
        assert_eq!(chars.next(), Some(','));
        assert_eq!(parse_number(&mut chars), 1766319049);
        assert_eq!(chars.next(), None);

        struct Flaky(bool);
        impl Iterator for Flaky {
            type Item = &'static str;
            fn next(&mut self) -> Option<Self::Item> {
                if self.0 {
                    self.0 = false;
                    Some("totally the last item")
                } else {
                    self.0 = true; // D'oh!
                    None
                }
            }
        }
        let mut flaky = Flaky(true);
        assert_eq!(flaky.next(), Some("totally the last item"));
        assert_eq!(flaky.next(), None);
        assert_eq!(flaky.next(), Some("totally the last item"));
        let mut not_flaky = Flaky(true).fuse();
        assert_eq!(not_flaky.next(), Some("totally the last item"));
        assert_eq!(not_flaky.next(), None);
        assert_eq!(not_flaky.next(), None);
        #[derive(Debug)]
        struct MyStruct(String, String);
        /*struct MyStruct {
            val1: (String, String),
            val2: (i32, i32),
        }*/
        let val = MyStruct("1".to_string(), "2".to_string());
        dbg!(&val);
        println!("{:#?}", &val);

        let bee_parts = ["head", "thorax", "abdomen"];
        let mut iter = bee_parts.into_iter();
        assert_eq!(iter.next(), Some("head"));
        assert_eq!(iter.next_back(), Some("abdomen"));
        assert_eq!(iter.next(), Some("thorax"));
        assert_eq!(iter.next_back(), None);
        assert_eq!(iter.next(), None);
        let meals = ["breakfast", "lunch", "dinner"];
        let mut iter = meals.iter().rev();
        assert_eq!(iter.next(), Some(&"dinner"));
        assert_eq!(iter.next(), Some(&"lunch"));
        assert_eq!(iter.next(), Some(&"breakfast"));
        assert_eq!(iter.next(), None);

        let upper_case: String = "große"
            .chars()
            .inspect(|c| println!("before: {:?}", c))
            //.flat_map(char::to_uppercase)
            .flat_map(|x| x.to_uppercase())
            .inspect(|c| println!("after: {:?}", c))
            .collect();
        let endings = vec!["once", "twice", "chicken soup with rice"];
        let val = endings
            .iter()
            .inspect(|c| println!("{:#?}", c))
            .map(|x| x.to_uppercase())
            .collect::<Vec<_>>();
        dbg!(val);
        let v: Vec<_> = (1..4).chain(vec![20, 30, 40]).collect();
        dbg!(v);

        let v: Vec<_> = (0..).zip("ABCD".chars()).collect();
        dbg!(v);
        let v: Vec<_> = std::ops::Range {
            start: 0,
            end: u32::MAX,
        }
        .zip("ABCD".chars())
        .collect();
        dbg!(v);

        use std::iter::repeat;
        let endings = vec!["once", "twice", "chicken soup with rice"];
        let rhyme: Vec<_> = repeat("going").zip(endings).collect();
        dbg!(rhyme);

        let mut lines = message.lines();
        println!("Headers:");
        for header in lines.by_ref().take_while(|l| !l.is_empty()) {
            println!("{}", header);
        }
        println!("\nBody:");
        for body in lines {
            println!("{}", body);
        }

        let mut it = "1\n2".lines().into_iter();
        it.by_ref()
            .inspect(|&e| println!("-{}", e))
            .take(1)
            .all(|_| true); // -1
        it.by_ref().inspect(|&e| println!("--{}", e)).all(|_| true); // --2
        let lines: Vec<_> = "1\n2".lines().into_iter().collect();
        lines.iter().all(|&l| {
            println!("{}", l);
            true
        });
        lines.iter().all(|&l| {
            println!("{}", l);
            true
        });

        let a = ['1', '2', '3', '='];
        assert_eq!(a.iter().next(), Some(&'1'));
        assert_eq!(a.iter().cloned().next(), Some('1'));

        let dirs = ["North", "East", "South", "West"];
        let mut spin = dirs.iter().cycle();
        assert_eq!(spin.next(), Some(&"North"));
        assert_eq!(spin.next(), Some(&"East"));
        assert_eq!(spin.next(), Some(&"South"));
        assert_eq!(spin.next(), Some(&"West"));
        assert_eq!(spin.next(), Some(&"North"));
        assert_eq!(spin.next(), Some(&"East"));
    }
    return;
    {
        let text = " ponies \n giraffes\niguanas \nsquid";
        let l: std::collections::LinkedList<String> = text
            .lines()
            .map(str::trim)
            .map(String::from)
            .filter(|s| s != "iguanas")
            .collect();
        dbg!(l);
        let text = "1\nfrond .25 289\n3.1415 estuary\n";
        use std::str::FromStr;
        for number in text
            .split_whitespace()
            .filter_map(|w| f64::from_str(w).ok())
        {
            println!("{:4.2}", number.sqrt());
        }
        use std::collections::HashMap;
        let mut major_cities = HashMap::new();
        major_cities.insert("Japan", vec!["Tokyo", "Kyoto"]);
        major_cities.insert("The United States", vec!["Portland", "Nashville"]);
        major_cities.insert("Brazil", vec!["São Paulo", "Brasilia"]);
        major_cities.insert("Kenya", vec!["Nairobi", "Mombasa"]);
        major_cities.insert("The Netherlands", vec!["Amsterdam", "Utrecht"]);
        let countries = ["Japan", "Brazil", "Kenya", "China"];
        let empty_vec = Vec::<&str>::new();
        for &city in countries.into_iter().flat_map(|country| {
            // &major_cities[country] // panic since China is not in map
            if major_cities.contains_key(country) {
                return &major_cities[country];
            }
            &empty_vec
        }) {
            println!("{}", city);
        }
        let v = ["1", "2"];
        let mut map: HashMap<&str, Vec<&str>> = HashMap::new();
        map.insert("1", vec!["10", "11"]);
        map.insert("2", vec!["20", "21"]);
        for element in &map {
            element.1.iter().all(|x| {
                print!("{}", x);
                true
            });
            println!("");
        }
        println!("{}", map.len());
        for element in v.iter().flat_map(|x| &map[x]) {
            println!("{}", element);
        }
        println!("{}", map.len());
        let v = "test".to_string();
        let v_chars = v.chars();
        dbg!(v_chars);
        let v_u: Vec<char> = v.chars().flat_map(char::to_uppercase).collect();
        dbg!(v_u);
        let v_u: String = v.chars().flat_map(char::to_uppercase).collect();
        dbg!(v_u);
        let val = vec![None, Some("day"), None, Some("one")]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        dbg!(val);
        let val = vec![["1"], ["2"]].into_iter().flatten().collect::<Vec<_>>();
        dbg!(val);
    }
    return;
    {
        let v = vec!["antimony", "arsenic", "alumium", "selenium"];
        let mut iterator = v.into_iter();
        while let Some(element) = iterator.next() {
            println!("{}", element);
        }

        let v = vec![4, 20, 12, 8, 6];
        let mut iterator = v.iter();
        assert_eq!(iterator.next(), Some(&4));
        assert_eq!(iterator.next(), Some(&20));
        assert_eq!(iterator.next(), Some(&12));
        assert_eq!(iterator.next(), Some(&8));
        assert_eq!(iterator.next(), Some(&6));
        assert_eq!(iterator.next(), None);

        use std::ffi::OsStr;
        use std::path::Path;
        let path = Path::new("C:/Users/JimB/Downloads/Fedora.iso");
        let mut iterator = path.iter();
        assert_eq!(iterator.next(), Some(OsStr::new("C:")));
        assert_eq!(iterator.next(), Some(OsStr::new("Users")));
        assert_eq!(iterator.next(), Some(OsStr::new("JimB")));

        use std::collections::BTreeSet;
        let mut favorites = BTreeSet::new();
        favorites.insert("Lucy in the Sky With Diamonds".to_string());
        favorites.insert("Liebesträume No. 3".to_string());
        let mut it = favorites.into_iter();
        assert_eq!(it.next(), Some("Liebesträume No. 3".to_string()));
        assert_eq!(it.next(), Some("Lucy in the Sky With Diamonds".to_string()));
        assert_eq!(it.next(), None);

        let v = vec![4, 20, 12, 8, 6];
        let mut iterator = v.into_iter();
        assert_eq!(iterator.next(), Some(4));
        let v = vec![4, 20, 12, 8, 6];
        let mut iterator = (&v).into_iter();
        assert_eq!(iterator.next(), Some(&4));
        let mut v = vec![4, 20, 12, 8, 6];
        let mut iterator = (&mut v).into_iter();
        assert_eq!(iterator.next(), Some(&mut 4));

        use std::fmt::Debug;
        fn dump<T, U>(t: T)
        where
            T: IntoIterator<Item = U>,
            U: Debug,
        {
            for u in t {
                println!("{:?}", u);
            }
        }

        let lengths: Vec<f64> =
            std::iter::from_fn(|| Some((rand::random::<f64>() - rand::random::<f64>()).abs()))
                .take(1000)
                .collect();
        dbg!(lengths);
        use num::Complex;
        fn escape_time(c: Complex<f64>, limit: usize) -> Option<usize> {
            let zero = Complex { re: 0.0, im: 0.0 };
            std::iter::successors(Some(zero), |&z| Some(z * z + c))
                .take(limit)
                .enumerate()
                .find(|(_i, z)| z.norm_sqr() > 4.0)
                .map(|(i, _z)| i)
        }
        fn fibonacci() -> impl Iterator<Item = usize> {
            let mut state = (0, 1);
            std::iter::from_fn(move || {
                state = (state.1, state.0 + state.1);
                Some(state.0)
            })
        }
        use std::iter::FromIterator;
        let mut outer = "Earth".to_string();
        let inner = String::from_iter(outer.drain(1..4));
        assert_eq!(outer, "Eh");
        assert_eq!(inner, "art");
    }
    return;
    {
        let val_str = "test string";
        let val_byte_vec: Vec<u8> = val_str.bytes().into_iter().collect();
        let val_char_vec: Vec<char> = val_str.chars().into_iter().collect();
        dbg!(val_byte_vec);
        dbg!(val_char_vec);

        let val_str: std::borrow::Cow<str> = format!("value is {}", 1.00f64).into();
        let mut val_str_vec: Vec<String> = Vec::new();
        val_str_vec.push(val_str.into());
    }
    return;
    {
        use std::collections::HashMap;
        #[derive(Debug)]
        struct Request {
            method: String,
            url: String,
            headers: HashMap<String, String>,
            body: Vec<u8>,
        }
        #[derive(Debug)]
        struct Response {
            code: u32,
            headers: HashMap<String, String>,
            body: Vec<u8>,
        }
        type BoxedCallback = Box<dyn Fn(&Request) -> Response>;
        struct BasicRouter {
            routes: HashMap<String, BoxedCallback>,
        }
        impl BasicRouter {
            fn new() -> BasicRouter {
                BasicRouter {
                    routes: HashMap::new(),
                }
            }
            fn add_route<C>(&mut self, route: &str, f: C)
            where
                C: Fn(&Request) -> Response + 'static,
            {
                self.routes.insert(route.to_string(), Box::new(f));
            }
        }
        impl BasicRouter {
            fn handle_request(&self, r: &Request) -> Response {
                match self.routes.get(&r.url) {
                    None => Response {
                        code: 404,
                        headers: HashMap::new(),
                        body: Vec::new(),
                    },
                    Some(c) => c(r),
                }
            }
        }
        fn root_response(r: &Request) -> Response {
            let headers = HashMap::new();
            Response {
                code: 100,
                headers,
                body: "/".into(),
            }
        }
        fn test_response(r: &Request) -> Response {
            let headers = HashMap::new();
            Response {
                code: 100,
                headers,
                body: "/test".into(),
            }
        }
        let mut router = BasicRouter::new();
        router.add_route("/", root_response);
        router.add_route("/test", test_response);
        let req = Request {
            method: "GET".to_string(),
            url: "/".to_string(),
            headers: HashMap::new(),
            body: Vec::new(),
        };
        let resp = router.handle_request(&req);
        dbg!(req);
        dbg!(resp);

        type CallBack = fn(&Request) -> Response;
        struct NewRouter {
            routes: HashMap<String, CallBack>,
        }
        impl NewRouter {
            fn new() -> NewRouter {
                NewRouter {
                    routes: HashMap::new(),
                }
            }
            fn add_route(&mut self, route: &str, c: CallBack) {
                self.routes.insert(route.to_string(), c);
            }
            fn handle_request(&self, r: &Request) -> Response {
                match self.routes.get(&r.url) {
                    Some(c) => c(r),
                    None => Response {
                        code: 404,
                        headers: HashMap::new(),
                        body: Vec::new(),
                    },
                }
            }
        }
        let mut router = NewRouter {
            routes: HashMap::new(),
        };
        // let mut router = NewRouter::new();
        router.add_route("/", root_response);
        router.add_route("/new", |r: &Request| -> Response {
            Response {
                code: 100,
                headers: HashMap::new(),
                body: "new".into(),
            }
        });
        let req = Request {
            method: "GET".into(),
            url: "/new".into(),
            headers: [
                ("1".to_string(), "1".to_string()),
                ("2".to_string(), "2".to_string()),
            ]
            .into_iter()
            .collect(),
            body: Vec::new(),
        };
        let resp = router.handle_request(&req);
        dbg!(req);
        dbg!(resp);
    }
    return;
    {
        let haystack: Vec<String> = vec!["some", "long", "list", "of", "strings"]
            .into_iter()
            .map(String::from)
            .collect();

        struct City {
            name: String,
            population: i64,
            country: String,
        }
        fn city_population_descending(city: &City) -> i64 {
            -city.population
        }
        fn sort_cities(cities: &mut Vec<City>) {
            cities.sort_by_key(city_population_descending);
        }
        use std::thread;
        fn start_sorting_thread(
            mut cities: Vec<City>,
            criterias: Vec<i64>,
        ) -> thread::JoinHandle<Vec<City>> {
            let key_fn = move |city: &City| -> i64 {
                // if let Some(_) = criterias.into_iter().find(|x| *x == city.population) {
                //     return 0;
                // }
                if criterias.contains(&city.population) {
                    return 0;
                }
                if let Some(_) = criterias.iter().find(|&x| *x == city.population) {
                    return 0;
                }
                -city.population
            };
            thread::spawn(move || {
                cities.sort_by_key(key_fn);
                cities
            })
        }
        fn start_sorting_thread_new(
            mut cities: Vec<City>,
            criterias: Vec<i64>,
        ) -> thread::JoinHandle<Vec<City>> {
            let mut iters = criterias.into_iter();
            let key_fn = move |city: &City| -> i64 {
                if let Some(_) = iters.find(|x| *x == city.population) {
                    return 0;
                }
                -city.population
            };
            thread::spawn(move || {
                cities.sort_by_key(key_fn);
                cities
            })
        }
        fn count_selected_cities(cities: &Vec<City>, test_fn: fn(&City) -> bool) -> usize {
            let mut count = 0;
            for city in cities {
                if test_fn(city) {
                    count += 1;
                }
            }
            count
        }
        fn has_monster_attacks(city: &City) -> bool {
            city.population > 1000000
        }
        let my_cities = vec![
            City {
                name: "a".to_string(),
                population: 100000,
                country: "a".to_string(),
            },
            City {
                name: "b".to_string(),
                population: 1010000,
                country: "b".to_string(),
            },
        ];
        let n = count_selected_cities(&my_cities, has_monster_attacks);
        dbg!(n);
        fn count_selected_cities_new<F>(cities: &Vec<City>, test_fn: F) -> usize
        where
            F: Fn(&City) -> bool,
        {
            let mut count = 0;
            for city in cities {
                if test_fn(city) {
                    count += 1;
                }
            }
            count
        }
        let n = count_selected_cities(&my_cities, |city: &City| city.population > 10000);
        dbg!(n);
        let n = count_selected_cities_new(&my_cities, |city: &City| city.population > 10000);
        dbg!(n);

        let dict: Vec<String> = vec!["1", "2", "3", "4"]
            .into_iter()
            .map(String::from)
            .collect();
        let debug_dump_dict = || {
            for value in &dict {
                // no need to move
                println!("{:?}", value);
            }
        };
        fn call_twice<F>(f: F) -> ()
        where
            F: Fn() -> (),
        {
            f();
            f();
        }
        call_twice(debug_dump_dict);
        let mut i = 0;
        let incr = || {
            i += 1; // incr借用了 i的一个可变引用
            println!("Ding! i is now: {}", i);
        };
        fn call_twice_new<F>(mut f: F)
        where
            F: FnMut(),
        {
            f();
            f();
        }
        call_twice_new(incr);

        let y = 10;
        let add_y = |x| x + y;
        let copy_of_add_y = add_y; // 这个闭包是`Copy`，因此...
        assert_eq!(add_y(copy_of_add_y(22)), 42); // ...我们可以使用这两个
        let mut x = 0;
        let mut add_to_x = |n: i32| {
            x += n;
            x
        };
        let copy_of_add_to_x = add_to_x; // 移动，而不是拷贝
                                         // assert_eq!(add_to_x(copy_of_add_to_x(1)), 2); // 错误：使用了被移动的值
        let mut greeting = "Hello, ".to_string();
        let mut greet = move |name| {
            greeting.push_str(name);
            println!("{}", greeting);
        };
        greet("Test");
        greet.clone()("Alfred");
        greet.clone()("Bruce");
    }
    return;
    {
        use std::borrow::Cow;
        enum Error {
            OutOfMemory,
            StackOverflow,
            MachineOnFire,
            Unfathomable,
            FileNotFound(String),
        }
        fn describe(error: &Error) -> Cow<'static, str> {
            match *error {
                Error::OutOfMemory => "out of memory".into(),
                Error::StackOverflow => "stack overflow".into(),
                Error::MachineOnFire => "machine on fire".into(),
                Error::Unfathomable => "machine bewildered".into(),
                Error::FileNotFound(ref path) => format!("file not found: {}", path).into(),
            }
        }
        let err = Error::MachineOnFire;
        println!("Disaster has struct: {}", describe(&err));
        let mut log = Vec::<String>::new();
        log.push(describe(&err).into_owned());
    }
    return;
    {
        let huge = 2_000_000_000_000i64;
        let smaller = huge as i32;
        println!("{}", smaller);
        use std::convert::TryInto;
        let smaller: i32 = huge
            .try_into()
            .unwrap_or_else(|e: std::num::TryFromIntError| {
                println!("error: {}", e.to_string());
                if huge >= 0 {
                    println!("i32::MAX");
                    i32::MAX
                } else {
                    println!("i32::MIN");
                    i32::MIN
                }
            });
        dbg!(smaller);
        struct Transform {
            val: i32,
        }
        #[derive(Debug)]
        enum TransformErr {
            BAD_GT_0,
            BAD_LT_0,
        }
        #[derive(Debug)]
        struct TransformResult {
            result: i32,
        };
        impl std::convert::TryInto<TransformResult> for Transform {
            type Error = TransformErr;
            fn try_into(self) -> Result<TransformResult, Self::Error> {
                if self.val == 0 {
                    return Ok(TransformResult { result: 0 });
                } else if self.val < 0 {
                    return Err(TransformErr::BAD_LT_0);
                }
                Err(TransformErr::BAD_GT_0)
            }
        }
        let err_handler = |e: TransformErr| -> TransformResult {
            println!("error: {:?}", e);
            TransformResult { result: -1 }
        };
        let transform = Transform { val: 11 };
        let transform_result: TransformResult = transform.try_into().unwrap_or_else(err_handler);
        let transform = Transform { val: 0 };
        let transform_result: TransformResult = transform.try_into().unwrap_or_else(err_handler);
        dbg!(transform_result);
    }
    return;
    {
        struct RcBox<T: ?Sized> {
            ref_count: usize,
            value: T,
        }
        let boxed_lunch: RcBox<String> = RcBox {
            ref_count: 1,
            value: "lunch".to_string(),
        };
        let boxed_displayable: &RcBox<dyn std::fmt::Display> = &boxed_lunch;
        fn display(boxed: &RcBox<dyn std::fmt::Display>) {
            println!("For your enjoyment: {}", &boxed.value);
        }
        display(boxed_displayable);
        struct Selector<T> {
            elements: Vec<T>,
            current: usize,
        }
        use std::ops::{Deref, DerefMut};
        impl<T> Deref for Selector<T> {
            type Target = T;
            fn deref(&self) -> &T {
                &self.elements[self.current]
            }
        }
        impl<T> DerefMut for Selector<T> {
            fn deref_mut(&mut self) -> &mut T {
                &mut self.elements[self.current]
            }
        }
        let mut s = Selector {
            elements: vec!['x', 'y', 'z'],
            current: 2,
        };
        assert_eq!(*s, 'z');
        assert!(s.is_alphabetic());
        *s = 'w';
        assert_eq!(s.elements, ['x', 'y', 'w']);
        fn show_it(thing: &str) {
            println!("{}", thing);
        }
        let s = Selector {
            elements: vec!["good", "bad", "ugly"],
            current: 2,
        };
        show_it(*s);
        use std::fmt::Display;
        fn show_it_generic<T: Display>(thing: T) {
            println!("{}", thing);
        }
        show_it_generic(*s);
        use std::collections::HashSet;
        let squares = [4, 9, 16, 25, 36, 49, 64];
        let (set1, set2): (HashSet<i32>, HashSet<i32>) = squares.iter().partition(|n| **n < 25);
        dbg!(set1);
        dbg!(set2);
        let (upper, lower): (String, String) = "Great Teacher Onizuka"
            .chars()
            .partition(|&c| c.is_uppercase());
        assert_eq!(upper, "GTO");
        assert_eq!(lower, "reat eacher nizuka");
    }
    return;
    {
        struct Image<T> {
            width: usize,
            height: usize,
            pixels: Vec<T>,
        }
        impl<T: Default + Copy> Image<T> {
            fn new(width: usize, height: usize) -> Image<T> {
                Image {
                    width: width,
                    height: height,
                    pixels: vec![T::default(); width * height],
                }
            }
        }
        impl<T> std::ops::Index<usize> for Image<T> {
            type Output = [T];
            fn index(&self, row: usize) -> &[T] {
                let start = row * self.width;
                &self.pixels[start..start + self.width]
            }
        }
        impl<T> std::ops::IndexMut<usize> for Image<T> {
            fn index_mut(&mut self, row: usize) -> &mut [T] {
                let start = row * self.width;
                &mut self.pixels[start..start + self.width]
            }
        }
    }
    return;
    {
        #[derive(Clone, Copy, Debug)]
        struct Complex<T> {
            re: T,
            im: T,
        }
        impl<T> std::ops::Add<Complex<T>> for Complex<T>
        where
            T: std::ops::Add<Output = T>,
        {
            type Output = Complex<T>;
            fn add(self, rhs: Self) -> Self::Output {
                Complex {
                    re: self.re + rhs.re,
                    im: self.im + rhs.im,
                }
            }
        }
        #[derive(Debug, PartialEq)]
        struct Interval<T> {
            lower: T, // 包括
            upper: T, // 不包括
        }
        use std::cmp::{Ordering, PartialOrd};
        impl<T> PartialOrd<Interval<T>> for Interval<T>
        where
            T: PartialOrd,
        {
            fn partial_cmp(&self, other: &Interval<T>) -> Option<Ordering> {
                if self == other {
                    Some(Ordering::Equal)
                } else if self.lower >= other.upper {
                    Some(Ordering::Greater)
                } else if self.upper <= other.lower {
                    Some(Ordering::Less)
                } else {
                    None
                }
            }
        }
        let mut vec = Vec::<Interval<i32>>::new();
        vec.push(Interval::<i32> { lower: 1, upper: 4 });
        vec.push(Interval::<i32> { lower: 2, upper: 3 });
        vec.push(Interval::<i32> { lower: 3, upper: 2 });
        vec.push(Interval::<i32> { lower: 4, upper: 1 });
        dbg!(&vec);
        vec.sort_by_key(|v| v.upper);
        dbg!(&vec);
        vec.sort_by_key(|v| std::cmp::Reverse(v.upper));
        dbg!(&vec);
    }
    return;
    {
        struct MyStruct {
            a: String,
            b: String,
        }
        impl std::clone::Clone for MyStruct {
            fn clone(&self) -> Self {
                MyStruct {
                    a: self.a.clone(),
                    b: self.b.clone(),
                }
            }
        }
        trait Mytrait {
            fn compare(&self, val: &Self) -> i32;
            fn compare2(val1: &Self, val2: &Self) -> i32;
        }
        impl Mytrait for MyStruct {
            fn compare(&self, val: &Self) -> i32 {
                if self.a == val.a && self.b == val.b {
                    return 0;
                }
                -1
            }
            fn compare2(val1: &Self, val2: &Self) -> i32 {
                if val1.a == val2.a && val1.b == val2.b {
                    return 0;
                }
                -1
            }
        }
        trait StringSet {
            fn new() -> Self;
            fn from_slice(strings: &[&str]) -> Self;
            fn contains(&self, string: &str) -> bool;
            fn add(&mut self, string: &str);
        }
        fn unknown_words<S: StringSet>(document: &[String], wordlist: &S) -> S {
            let mut unknowns = S::new();
            for word in document {
                if !wordlist.contains(word) {
                    unknowns.add(word)
                }
            }
            unknowns
        }
        // trait StringSetNew {
        //     fn new() -> Self
        //     where
        //         Self: Sized;
        //     fn from_slice(strings: &[&str]) -> Self
        //     where
        //         Self: Sized;
        //     fn contains(&self, string: &str) -> bool;
        //     fn add(&mut self, string: &str);
        // }
        fn dump<I>(iter: &mut I)
        where
            I: Iterator,
            I::Item: std::fmt::Debug,
        {
            for (ref index, ref s) in iter.enumerate() {
                println!("{}: {:?}", index, s);
            }
        }
        fn dump_string(iter: &mut dyn Iterator<Item = String>) {
            for (index, s) in iter.enumerate() {
                println!("{}: {:?}", index, s);
            }
        }
        use std::ops::{Add, Mul};
        fn do_calc<N>(val1: &N, val2: &N) -> N
        where
            N: Add<Output = N> + Mul<Output = N> + Default + Clone,
        {
            let mut val = N::default();
            val = val1.clone() + val2.clone() + val1.clone() * val2.clone();
            val
        }
    }
    return;
    {
        use serde::Serialize;
        use serde_json;
        use std::collections::HashMap;
        use std::io::Write;
        pub fn write_json(map: &HashMap<String, String>) -> std::io::Result<()> {
            let mut file = std::fs::File::create("./json")?;
            let mut serializer = serde_json::Serializer::new(file);
            map.serialize(&mut serializer)?;
            Ok(())
        }
        let mut map = HashMap::<String, String>::new();
        map.insert("1".to_string(), "one".to_string());
        map.insert("2".to_string(), "two".to_string());
        map.insert("3".to_string(), "three".to_string());
        map.insert("4".to_string(), "four".to_string());
        write_json(&map).expect("write json failed");
    }
    return;
    {
        let mut val_str = String::new();
        val_str.insert_str(0, "hello");
        let pos = val_str.len();
        val_str.insert_str(pos, " world");
        dbg!(val_str);
    }
    return;
    {
        let mut buf = std::vec::Vec::new();
        let writer: &mut dyn std::io::Write;
        writer = &mut buf;
        let val = std::io::Result::Ok("test");
        fn compare<T>(val1: &T, val2: &T) -> i32
        where
            T: Ord + Eq,
        {
            if val1 == val2 {
                0
            } else if val1 < val2 {
                -1
            } else {
                1
            }
        }
        trait Vegetable {}
        struct Salad {
            veggies: Vec<Box<dyn Vegetable>>,
        }
        trait Meat {}
        let val_ref: &dyn Vegetable;
        struct Mywriter {};
        impl std::io::Write for Mywriter {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                println!("write {}", buf.len());
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                dbg!("flush");
                Ok(())
            }
        }
        use std::io::Write;
        let mut my_writer = Mywriter {};
        my_writer.write(b"test").expect("write to Mywriter failed");

        struct HtmlDocument {
            content: String,
        }
        trait WriteHtml {
            fn write_html(&mut self, doc: &HtmlDocument) -> std::io::Result<()>;
        }
        impl<W: std::io::Write> WriteHtml for W {
            fn write_html(&mut self, doc: &HtmlDocument) -> std::io::Result<()> {
                self.write(&"<HTML>".to_string().as_bytes())?;
                self.write(&doc.content.as_bytes())?;
                self.write(&"</HTML>".to_string().as_bytes())?;
                Ok(())
            }
        }
        let mut html_content = Vec::new();
        let html_document = HtmlDocument {
            content: "test html".to_string(),
        };
        html_content.write_html(&html_document);
        use std::str;
        let html_content_str = str::from_utf8(&html_content)
            .expect("convert to str failed!")
            .to_string();
        dbg!(html_content_str);
    }
    return;
    {
        fn output_txt(out: &mut dyn std::io::Write, txt: &[u8]) -> GenericResult<()> {
            out.write_all(txt)?;
            Ok(())
        }
        let mut out_buf = Vec::new();
        let result = output_txt(&mut out_buf, b"hello Rust!");
        match result {
            Err(ref err) => {
                eprintln!("error {}", err.to_string());
            }
            Ok(..) => {}
        }
        if let Ok(_) = result.as_ref() {
            println!("ok {:?}", &out_buf);
        } else {
            if let Some(ref err) = result.as_ref().err() {
                eprintln!("error {}", err.to_string());
            }
        }
        let result_ref = result.as_ref();
        match result_ref {
            Err(err) => {
                eprintln!("error {}", err.to_string());
            }
            Ok(..) => {}
        }
        if let Ok(_) = result_ref {
            println!("ok {:?}", &out_buf);
        } else {
            if let Some(err) = result_ref.err() {
                eprintln!("error {}", err.to_string());
            }
        }
    }
    return;
    {
        let r;
        {
            let x = 1;
            r = &x;
        }
        // assert_eq!(*r, 1); // bad: reads memory `x` used to occupy
        fn smallest<'life>(v: &'life [i32]) -> &i32 {
            let mut s = &v[0];
            for r in &v[1..] {
                if r < s {
                    s = r;
                }
            }
            s
        }
        let v = vec![1, 2, 3, 4, 5];
        let _v = smallest(&v);
        dbg!(_v);
    }
    {
        fn max<'life>(x: &'life i32, y: &'life i32) -> &'life i32 {
            if x > y {
                x
            } else {
                y
            }
        }
        // return &i32, so need to specify lifetime
        fn max_i32<'life>(x: &'life i32, y: &'life i32) -> &'life i32 {
            if x > y {
                x
            } else {
                x
            }
        }
        fn smallest<'life>(v1: &'life [i32], v2: &'life [i32]) -> &'life i32 {
            if v1[0] < v2[0] {
                &v1[0]
            } else {
                &v2[0]
            }
        }
        let v = max(&1, &2);
        dbg!(v);
    }
    return;
    {
        #[derive(Debug)]
        enum BinaryTree<T> {
            Empty,
            NonEmpty(Box<TreeNode<T>>),
        }
        #[derive(Debug)]
        struct TreeNode<T> {
            element: T,
            left: BinaryTree<T>,
            right: BinaryTree<T>,
        }
        impl<T: Ord + Clone> BinaryTree<T> {
            fn add(&mut self, val: &T) {
                match *self {
                    Empty => {
                        *self = NonEmpty(Box::new(TreeNode {
                            element: val.clone(),
                            left: Empty,
                            right: Empty,
                        }));
                    }
                    NonEmpty(ref mut current) => {
                        if val < &(current.element) {
                            current.left.add(val);
                        } else {
                            current.right.add(val);
                        }
                    }
                }
            }
        }
        use BinaryTree::*;
        let jupiter_tree = NonEmpty(Box::new(TreeNode {
            element: "Jupiter",
            left: Empty,
            right: Empty,
        }));
        let mercury_tree = NonEmpty(Box::new(TreeNode {
            element: "Mercury",
            left: Empty,
            right: Empty,
        }));
        let mars_tree = NonEmpty(Box::new(TreeNode {
            element: "Mars",
            left: jupiter_tree,
            right: mercury_tree,
        }));
        let tree = NonEmpty(Box::new(TreeNode {
            element: "Saturn",
            left: mars_tree,
            right: Empty,
        }));
        println!("tree {:?}", &tree);
        let mut tree = BinaryTree::<String>::Empty;
        tree.add(&"2".to_string());
        println!("tree {:?}", &tree);
        tree.add(&"1".to_string());
        println!("tree {:?}", &tree);
        tree.add(&"3".to_string());
        println!("tree {:?}", &tree);
    }
    return;
    {
        fn get_point_position(x: &i32, y: &i32) -> &'static str {
            use std::cmp::Ordering;
            match (x.cmp(&0), y.cmp(&0)) {
                (Ordering::Equal, Ordering::Equal) => "on center",
                (Ordering::Equal, _) => "on x-axis",
                (_, Ordering::Equal) => "on y-axis",
                _ => "somewhere else",
            }
        }
        let x = 10;
        let y = 10;
        println!("{} and {} position {}", x, y, get_point_position(&x, &y));
        struct MyStruct {
            a: i32,
            b: i32,
            c: i32,
            d: i32,
        }
        let val = MyStruct {
            a: 0,
            b: 0,
            c: 0,
            d: 0,
        };
        match &val {
            MyStruct { a: 1, .. } => println!("a is correct"),
            MyStruct { b: 2, .. } => println!("b is correct"),
            MyStruct { c: 3, .. } => println!("c is correct"),
            MyStruct { d: 4, .. } => println!("d is correct"),
            _ => println!("val all fields are wrong"),
        }
        match &val {
            MyStruct { b, c, .. } => println!("val b {} c {}", b, c),
        }
        println!("val a {} b {} c {} d {}", val.a, val.b, val.c, val.d);
        let vec = [1, 2, 3, 4];
        let vec_ref = &vec;
        let vec_str = ["1", "2", "3", "4"];
        let vec_str_ref: &[&str] = &vec_str;
    }
    return;
    {
        #[derive(Copy, Clone, Debug, PartialEq, Eq)]
        enum TimeUnit {
            Seconds,
            Minutes,
            Hours,
            Days,
            Months,
            Years,
        }
        impl TimeUnit {
            fn plural(self) -> &'static str {
                match self {
                    TimeUnit::Seconds => "seconds",
                    TimeUnit::Minutes => "minutes",
                    TimeUnit::Hours => "hours",
                    TimeUnit::Days => "days",
                    TimeUnit::Months => "months",
                    TimeUnit::Years => "years",
                }
            }
            fn singular(self) -> &'static str {
                self.plural().trim_end_matches('s')
            }
        }
        enum RoughTime {
            InThePast(TimeUnit, i32),
            JustNow,
            InTheFuture(TimeUnit, i32),
        }
        fn get_rt_str(rt: &RoughTime) -> String {
            match rt {
                RoughTime::JustNow => "now".to_string(),
                RoughTime::InThePast(unit, 1) => format!("1 {} ago", unit.singular()),
                RoughTime::InThePast(unit, count) => format!("{} {} ago", count, unit.plural()),
                RoughTime::InTheFuture(unit, 1) => format!("1 {} later", unit.singular()),
                RoughTime::InTheFuture(unit, count) => format!("{} {} later", count, unit.plural()),
            }
        }
        let rt_val = RoughTime::InThePast(TimeUnit::Months, 18);
        println!("{}", get_rt_str(&rt_val));
    }
    return;
    {
        use std::collections::HashMap;
        #[derive(Debug)]
        enum Json {
            Null,
            Boolean(bool),
            Number(f64),
            String(String),
            Array(Vec<Json>),
            Object(Box<HashMap<String, Json>>),
        }
        let json_val = Json::String("Test Json".to_string());
        println!("json_val {:?}", &json_val);
    }
    return;
    {
        let v = ["1", "2", "3"];
        let val = match do_test(1) {
            Ok(success_value) => success_value,
            Err(err) => return,
        };
        let fun = |x: i32| -> ! { loop {} };
    }
    return;
    {
        let strings: Vec<String> = vec![
            "1".to_string(),
            "2".to_string(),
            "3".to_string(),
            "4".to_string(),
        ];
        for rs in &strings {
            println!("String {:?} is at address {:p}.", rs, rs);
        }
    }
    return;
    {
        let x = 10;
        let y = 10;
        let rx = &x;
        let ry = &y;
        let rrx = &rx;
        let rry = &ry;
        assert!(rrx <= rry);
        assert!(rrx == rry);

        fn factorial(n: usize) -> usize {
            (1..n + 1).product()
        }
        let r = &factorial(6);
        assert_eq!(r + &1009, &1729 + 0);
    }
    return;
    {
        let string_replacen = "I like rust. Learning rust is my favorite!";
        let new_string_replacen = string_replacen.replacen("rust", "RUST", 1);
        dbg!(new_string_replacen);
        let mut s = "test".to_string();
        s.replace_range(0..2, "TE");
        dbg!(s);
    }
    return;
    {
        let regex_handler = match regex::Regex::new("[0-9]x[0-9]") {
            Err(_) => {
                return;
            }
            Ok(handler) => handler,
        };
        let result = regex_handler.replace("123x456", "X").to_string();
        println!("result {}", result);
        return;
    }
    let val = (123, String::from("123"));
    // let box_val = Box::<(i32, String)>::new(val);
    // let box_val : Box<(i32, String)> = Box::new(val);
    let mut primes = vec![2, 3, 5, 7];
    assert_eq!(primes.iter().product::<i32>(), 210);
    let mut test_vec = Vec::<u8>::new();
    test_vec.resize(100, 0);
    test_vec.reserve(1000);
    let mut new_test_vec = (0..100).collect::<Vec<u8>>();
    let languages: Vec<String> = std::env::args().skip(1).collect();
    for l in languages {
        println!(
            "{}: {}",
            l,
            if l.len() % 2 == 0 {
                "functional"
            } else {
                "imperative"
            }
        );
    }

    let v: Vec<f64> = vec![0.0, 0.707, 1.0, 0.707];
    let a: [f64; 4] = [0.0, -0.707, -1.0, -0.707];
    // let v_str = vec![String::from("a"), String::from("b"), String::from("c")];
    let v_str: Vec<String> = std::env::args().skip(1).collect();
    print_vec(&v_str);
    let str = "ABC";
    let method = br##"GE""T"##;
    println!("method {:?}", v);
    let noodles = "noodles".to_string();
    let oodles = &noodles[1..];
    let poodles = "某些卡纳达语字符";
    println!(
        "noodles len {} oodles len {} poodles len {} poodles chars count {}",
        noodles.len(),
        oodles.len(),
        poodles.len(),
        poodles.chars().count()
    );
    {
        let mut update_str = "test 1234";
        // update_str.make_ascii_uppercase();
        let sub_update_str = &update_str[5..];
        println!(
            "update_str: {} sub_update_str {}",
            update_str, sub_update_str
        );
    }
    {
        let mut update_str = "test 1234".to_string();
        // update_str[0] = ' ';
        println!("update_str: {}", update_str);
    }
    let fmt_str = format!("{}{}", "123", "456");
    println!("fmt_str: {}", fmt_str);
    assert!("peanut".contains("nut"));
    assert_eq!(" clean\n".trim(), "clean");
    for word in "veni, vidi, vici".split(", ") {
        assert!(word.starts_with("v"));
    }
    let fruits = ["mango", "apple", "banana", "litchi", "watermelon"];
    for a in fruits.iter() {
        print!("{} ", a);
    }
    println!("");

    let str1 = "str".to_string();
    let str2 = str1;
    test_ownership(str1);
    println!("value of str1 :{}", str1);

    struct Person {
        name: Option<String>,
        birth: i32,
    }
    let mut composers = Vec::new();
    composers.push(Person {
        name: Some("Palestrina".to_string()),
        birth: 1525,
    });
    let first_name = std::mem::replace(&mut composers[0].name, None);
    assert_eq!(first_name, Some("Palestrina".to_string()));
    assert_eq!(composers[0].name, None);
    {
        let mut v = vec!["a", "b", "c"];
        let rep_val = "R".to_string();
        let val = std::mem::replace(&mut v[1], &rep_val);
        println!("v replaced {:?} val {}", v, val);
    }
    {
        use std::rc::Rc;
        let s: Rc<String> = Rc::new("shirataki".to_string());
        let t: Rc<String> = s.clone();
        let u: Rc<String> = s.clone();
        assert!(s.contains("shira"));
        assert_eq!(t.find("taki"), Some(5));
        println!("{} are quite chewy, almost bouncy, but lack flavor", u);
    }
    return;
    {
        let do_test_ret = match do_test(1) {
            Ok(ret) => ret,
            _ => -1,
        };
        println!("do_test return {}", do_test_ret);
        do_test(0).expect("do_test error");
    }
}

mod other;
// fn test_run_cmd_line_process() {
//     other::run_cmd_line_process();
// }
mod async_chat;
