use image::png::PNGEncoder;
use image::ColorType;
use num::Complex;
use regex::Regex;
use std::num::TryFromIntError;
use std::{collections::LinkedList, fs::File};
use text_colorizer::Colorize;

type GenericError = Box<dyn std::error::Error + Send + Sync + 'static>;
type GenericResult<T> = Result<T, GenericError>;

// struct Complex<T> {
//     /// Real portion of the complex number
//     re: T,
//     /// Imaginary portion of the complex number
//     im: T,
// }

fn escape_time(c: Complex<f64>, limit: usize) -> Option<usize> {
    let mut z = Complex { re: 0.0, im: 0.0 };
    for i in 0..limit {
        if z.norm_sqr() > 4.0 {
            return Some(i);
        }
        z = z * z + c;
    }
    None
}

fn parse_pair<T: std::str::FromStr>(s: &str, separator: char) -> Option<(T, T)> {
    match s.find(separator) {
        None => None,
        Some(index) => match (T::from_str(&s[..index]), T::from_str(&s[index + 1..])) {
            (Ok(l), Ok(r)) => Some((l, r)),
            _ => None,
        },
    }
}

fn parse_complex(s: &str) -> Option<Complex<f64>> {
    match parse_pair::<f64>(s, ',') {
        Some((re_val, im_val)) => Some(Complex {
            re: re_val,
            im: im_val,
        }),
        None => None,
    }
}

fn pixel_to_point(
    bounds: (usize, usize),
    pixel: (usize, usize),
    upper_left: Complex<f64>,
    lower_right: Complex<f64>,
) -> Complex<f64> {
    let (width, height) = (
        lower_right.re - upper_left.re,
        upper_left.im - lower_right.im,
    );
    Complex {
        re: upper_left.re + pixel.0 as f64 * width / bounds.0 as f64,
        im: upper_left.im - pixel.1 as f64 * height / bounds.1 as f64,
        // Why subtraction here? pixel.1 increases as we go down,
        // but the imaginary component increases as we go up.
    }
}

fn render(
    pixels: &mut [u8],
    bounds: (usize, usize),
    upper_left: Complex<f64>,
    lower_right: Complex<f64>,
) {
    assert!(pixels.len() == bounds.0 * bounds.1);
    for row in 0..bounds.1 {
        for column in 0..bounds.0 {
            let point = pixel_to_point(bounds, (column, row), upper_left, lower_right);
            pixels[row * bounds.0 + column] = match escape_time(point, 255) {
                None => 0,
                Some(count) => 255 - count as u8,
            };
        }
    }
}

fn write_image(
    filename: &str,
    pixels: &[u8],
    bounds: (usize, usize),
) -> Result<(), std::io::Error> {
    let output = File::create(filename)?;
    let encoder = PNGEncoder::new(output);
    encoder.encode(
        &pixels,
        bounds.0 as u32,
        bounds.1 as u32,
        ColorType::Gray(8),
    )?;
    Ok(())
}

fn do_test_sub(val: i32) -> Result<i32, String> {
    let val1 = match File::create("test") {
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
    let regex_handler = Regex::new(pattern)?;
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

#[derive(Debug)]
struct Arguments {
    target: String,
    replacement: String,
    in_filename: String,
    out_filename: String,
}

fn print_usage(cmd_name: &str) {
    eprintln!(
        "{} - change occurrences of one string into another",
        cmd_name.green()
    );
    eprintln!(
        "Usage: {} <target> <replacement> <INPUT> <OUTPUT>",
        cmd_name.red().bold()
    );
}

fn parse_args() -> Result<Arguments, i32> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        print_usage(&args[0]);
        eprintln!(
            "{} wrong number of arguments: expected 4, got {}.",
            "Error:".red().bold(),
            args.len() - 1
        );
        return Err(-1);
    }
    Ok(Arguments {
        target: args[1].clone(),
        replacement: args[2].clone(),
        in_filename: args[3].clone(),
        out_filename: args[4].clone(),
    })
}

fn replace_string(content: &str, replace: &str, replace_with: &str) -> Result<String, i32> {
    let regex_handler = match Regex::new(replace) {
        Ok(handler) => handler,
        Err(error) => {
            eprintln!(
                "{}: {} bad regex: {:?}",
                "error".red().bold(),
                &replace,
                error
            );
            return Err(-1);
        }
    };
    Ok(regex_handler.replace_all(content, replace_with).to_string())
}

fn print_vec<T: std::fmt::Display>(n: &[T]) {
    for elt in n {
        println!("-- {} --", elt);
    }
}

fn test_ownership(strval: String) {
    println!("{}", strval);
}

use actix_web::{web, App, HttpResponse, HttpServer};

fn http_server_main() {
    let server = HttpServer::new(|| App::new().route("/", web::get().to(handle_get)));
    server
        .bind("0.0.0.0:8088")
        .expect("server bind error")
        .run()
        .expect("server run error");
}

fn handle_get() -> HttpResponse {
    HttpResponse::Ok().content_type("text/html").body(
        r####"
        <!DOCTYPE html>
        <html>
        <body>
        <h1>server is running</h1>
        <p>server status is running</p>
        </body>
        </html>
    "####,
    )
}

fn main() {
    {
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

        let val:Vec<_> = ["1", "2"].iter().map(|x| x.to_owned()).collect();
        dbg!(val);
    }
    return;
    {
        let text = " ponies \n giraffes\niguanas \nsquid";
        let l: LinkedList<String> = text
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

        use rand::random;
        let lengths: Vec<f64> =
            std::iter::from_fn(|| Some((random::<f64>() - random::<f64>()).abs()))
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
        use std::string::String;
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
        let smaller: i32 = huge.try_into().unwrap_or_else(|e: TryFromIntError| {
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
        let json_val = Json::String(("Test Json".to_string()));
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
        let regex_handler = match Regex::new("[0-9]x[0-9]") {
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
    {
        let args = match parse_args() {
            Ok(val) => val,
            Err(_error_code) => {
                std::process::exit(-1);
            }
        };
        println!("{:?}", args);
        let file_content = match std::fs::read_to_string(&args.in_filename) {
            Ok(data) => data,
            Err(error) => {
                eprintln!(
                    "{}: {} read_to_string: {:?}",
                    "error".red().bold(),
                    &args.in_filename,
                    error
                );
                std::process::exit(-1);
            }
        };
        let replaced_content = match replace_string(&file_content, &args.target, &args.replacement)
        {
            Ok(val) => val,
            Err(_error_code) => {
                std::process::exit(-1);
            }
        };
        match std::fs::write(&args.out_filename, &replaced_content) {
            Ok(_) => (),
            Err(error) => {
                eprintln!(
                    "{}: {} write: {:?}",
                    "error".red().bold(),
                    &args.out_filename,
                    error
                );
                std::process::exit(-1);
            }
        }
    }
    return;

    let _test_integer_var = 0;
    let _test_float_var = 3.14;
    let _test_vec_var = vec![0, 3];

    // let do_test_ret = match do_test(1) {
    //     Ok(ret) => ret,
    //     _ => -1
    // };
    // println!("do_test return {}", do_test_ret);
    // do_test(0).expect("do_test error");

    let mut args: Vec<String> = std::env::args().collect();
    // print_usage(&args[0]);
    if args.len() != 5 {
        eprintln!("Usage: {} FILE PIXELS UPPERLEFT LOWERRIGHT", args[0]);
        eprintln!(
            "Example: {} mandel.png 1000[xX]750 -1.20,0.35 -1,0.20",
            args[0]
        );
        std::process::exit(1);
    }
    // let pos: i32 = match args[2].find('x') {
    //     Some(pos) => pos as i32,
    //     _ => -1
    // };
    // if pos >= 0 {
    //     args[2] = args[2].replace("x", "X");
    // }
    //args[2] = args[2].replace("([0-9])+x([0-9])+", "\\1X\\2");
    args[2] = args[2].replace("x", "X");
    println!("{}", args[2]);
    let bounds = parse_pair::<usize>(&args[2], 'X').expect("error parsing image dimensions");
    let upper_left = parse_complex(&args[3]).expect("error parsing upper left corner point");
    let lower_right = parse_complex(&args[4]).expect("error parsing lower right corner point");
    let mut pixels = vec![0; bounds.0 * bounds.1];
    // render(&mut pixels, bounds, upper_left, lower_right);

    let threads = 8;
    let rows_per_band = bounds.1 / threads + 1;
    {
        let bands: Vec<&mut [u8]> = pixels.chunks_mut(rows_per_band * bounds.0).collect();
        crossbeam::scope(|spawner| {
            for (i, band) in bands.into_iter().enumerate() {
                let top = rows_per_band * i;
                let height = band.len() / bounds.0;
                let band_bounds = (bounds.0, height);
                let band_upper_left = pixel_to_point(bounds, (0, top), upper_left, lower_right);
                let band_lower_right =
                    pixel_to_point(bounds, (bounds.0, top + height), upper_left, lower_right);
                spawner.spawn(move |_| {
                    render(band, band_bounds, band_upper_left, band_lower_right);
                });
            }
        })
        .unwrap();
    }

    write_image(&args[1], &pixels, bounds).expect("error writing PNG file");
}

#[test]
fn test_parse_pair() {
    assert_eq!(parse_pair::<i32>("", ','), None);
    assert_eq!(parse_pair::<i32>("10,", ','), None);
    assert_eq!(parse_pair::<i32>(",10", ','), None);
    assert_eq!(parse_pair::<i32>("10,20", ','), Some((10, 20)));
    assert_eq!(parse_pair::<i32>("10,20xy", ','), None);
    assert_eq!(parse_pair::<f64>("0.5x", ','), None);
    assert_eq!(parse_pair::<f64>("0.5x1.5", 'x'), Some((0.5, 1.5)));
}

#[test]
fn test_parse_complex() {
    assert_eq!(
        parse_complex("1.25,-0.0625"),
        Some(Complex {
            re: 1.25,
            im: -0.0625
        })
    );
    assert_eq!(parse_complex(",-0.0625"), None);
}

#[test]
fn test_pixel_to_point() {
    assert_eq!(
        pixel_to_point(
            (100, 200),
            (25, 175),
            Complex { re: -1.0, im: 1.0 },
            Complex { re: 1.0, im: -1.0 }
        ),
        Complex {
            re: -0.5,
            im: -0.75
        }
    );
}
