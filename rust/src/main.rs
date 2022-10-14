use image::png::PNGEncoder;
use image::ColorType;
use num::Complex;
use regex::Regex;
use std::env;
use std::fs::read_to_string;
use std::fs::write;
use std::fs::File;
use std::result;
use std::str::FromStr;
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

fn parse_pair<T: FromStr>(s: &str, separator: char) -> Option<(T, T)> {
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
    let args: Vec<String> = env::args().collect();
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
        let html_content_str = str::from_utf8(&html_content).expect("convert to str failed!").to_string();
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
        // fn max_bad(x: &i32, y: &i32) -> &i32 {
        //     if x > y {
        //         x
        //     } else {
        //         x
        //     }
        // }
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
        // let jupiter_tree = NonEmpty(Box::new(TreeNode {
        //     element: "Jupiter",
        //     left: Empty,
        //     right: Empty,
        // }));
        // let mercury_tree = NonEmpty(Box::new(TreeNode {
        //     element: "Mercury",
        //     left: Empty,
        //     right: Empty,
        // }));
        // let mars_tree = NonEmpty(Box::new(TreeNode {
        //     element: "Mars",
        //     left: jupiter_tree,
        //     right: mercury_tree,
        // }));
        // let tree = NonEmpty(Box::new(TreeNode {
        //     element: "Saturn",
        //     left: mars_tree,
        //     right: Empty,
        // }));
        // println!("tree {:?}", &tree);
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
    // let regex_handler = match Regex::new("[0-9]x[0-9]") {
    //     Err(_) => {
    //         return;
    //     },
    //     Ok(handler) => handler
    // };
    // let result = regex_handler.replace("123x456", "X").to_string();
    // println!("result {}", result);
    // return;

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

    // let args = match parse_args() {
    //     Ok(val) => val,
    //     Err(_error_code) => {
    //         std::process::exit(-1);
    //     }
    // };
    // println!("{:?}", args);
    // let file_content = match read_to_string(&args.in_filename) {
    //     Ok(data) => data,
    //     Err(error) => {
    //         eprintln!(
    //             "{}: {} read_to_string: {:?}",
    //             "error".red().bold(),
    //             &args.in_filename,
    //             error
    //         );
    //         std::process::exit(-1);
    //     }
    // };
    // let replaced_content = match replace_string(&file_content, &args.target, &args.replacement) {
    //     Ok(val) => val,
    //     Err(_error_code) => {
    //         std::process::exit(-1);
    //     }
    // };
    // match write(&args.out_filename, &replaced_content) {
    //     Ok(_) => (),
    //     Err(error) => {
    //         eprintln!(
    //             "{}: {} write: {:?}",
    //             "error".red().bold(),
    //             &args.out_filename,
    //             error
    //         );
    //         std::process::exit(-1);
    //     }
    // }
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

    let mut args: Vec<String> = env::args().collect();
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
