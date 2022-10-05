use image::png::PNGEncoder;
use image::ColorType;
use num::Complex;
use regex::Regex;
use std::env;
use std::fs::read_to_string;
use std::fs::write;
use std::fs::File;
use std::str::FromStr;
use text_colorizer::Colorize;

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
    if val > 0 {
        return Ok(1);
    } else if val < 0 {
        return Ok(-1);
    }
    Err(String::from("cannot be 0"))
}

fn do_test(val: i32) -> Result<i32, String> {
    do_test_sub(val)?;
    Ok(100)
}

#[derive(Debug)]
struct Arguments {
    target: String,
    replacement: String,
    in_filename: String,
    out_filename: String,
}

fn print_usage(cmd_name: &String) {
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

fn replace_string(
    content: &String,
    replace: &String,
    replace_with: &String,
) -> Result<String, i32> {
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
fn main() {
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
    let box_val = Box::<(i32, String)>::new(val);
    let mut primes = vec![2, 3, 5, 7];
    assert_eq!(primes.iter().product::<i32>(), 210);
    let test_vec = Vec::<u8>::new();

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
