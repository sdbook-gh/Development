#![no_std]
#![no_main]
#![feature(core_intrinsics)]
#![feature(lang_items)]

#[allow(unused)]
#[panic_handler]
#[no_mangle]
pub fn panic(info: &core::panic::PanicInfo) -> ! {
    let mut cursor = Cursor {
        position: 0,
        foreground: Color::White,
        background: Color::Red,
    };
    for _ in 0..(80 * 25) {
        cursor.print(b" ");
    }
    cursor.position = 0;
    use core::fmt::Write;
    write!(cursor, "{}", info).unwrap();

    loop {
        x86_64::instructions::hlt();
    }
}

#[lang = "eh_personality"]
#[no_mangle]
pub extern "C" fn eh_personality() {}

#[derive(Clone, Copy)]
#[repr(u8)]
enum Color {
    Black = 0x0,
    White = 0xF,
    Blue = 0x1,
    BrightBlue = 0x9,
    Green = 0x2,
    BrightGreen = 0xA,
    Cyan = 0x3,
    BrightCyan = 0xB,
    Red = 0x4,
    BrightRed = 0xC,
    Magenta = 0x5,
    BrightMagenta = 0xD,
    Brown = 0x6,
    Yellow = 0xE,
    Gray = 0x7,
    DarkGray = 0x8,
}

struct Cursor {
    position: isize,
    foreground: Color,
    background: Color,
}

impl Cursor {
    fn color(&self) -> u8 {
        let fg = self.foreground as u8;
        let bg = (self.background as u8) << 4;
        fg | bg
    }
    fn print(&mut self, text: &[u8]) {
        let color = self.color();

        let framebuffer = 0xb8000 as *mut u8;

        for &character in text {
            unsafe {
                framebuffer.offset(self.position).write_volatile(character);
                framebuffer.offset(self.position + 1).write_volatile(color);
            }
            self.position += 2;
        }
    }
}
impl core::fmt::Write for Cursor {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        self.print(s.as_bytes());
        Ok(())
    }
}

#[no_mangle]
pub extern "C" fn _start() -> ! {
    let text = b"Hello Rust OS!";
    let mut cursor = Cursor {
        position: 0,
        foreground: Color::Red,
        background: Color::White,
    };
    cursor.print(text);

    let mut i = 0;
    while i < 100000000 {
        i += 1;
    }

    panic!("help!");
    loop {
        x86_64::instructions::hlt();
    }
}
