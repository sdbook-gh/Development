pub fn test_spin_loop() {
    use std::{thread, time};
    fn main() {
        for n in 1..1001 {
            let mut handlers: Vec<thread::JoinHandle<()>> = Vec::with_capacity(n);

            let start = time::Instant::now();
            for _m in 0..n {
                let handle = thread::spawn(|| {
                    let start = time::Instant::now();
                    let pause = time::Duration::from_millis(20);
                    while start.elapsed() < pause {
                        thread::yield_now();
                    }
                });
                handlers.push(handle);
            }

            while let Some(handle) = handlers.pop() {
                handle.join();
            }

            let finish = time::Instant::now();
            println!("{}\t{:02?}", n, finish.duration_since(start));
        }
    }
    main();
}

pub fn test_shared_variable() {
    use std::{thread, time};

    fn main() {
        let pause = time::Duration::from_millis(20);
        let handle1 = thread::spawn(move || {
            thread::sleep(pause);
        });
        let handle2 = thread::spawn(move || {
            thread::sleep(pause);
        });

        handle1.join();
        handle2.join();
    }
}

pub fn test_channel_simplex() {
    let (tx, rx) = crossbeam::channel::unbounded();
    std::thread::spawn(move || {
        tx.send("hello crossbeam!").expect("send failed");
    });
    crossbeam::select! {
        recv(rx) -> msg => {println!("{:?}", msg);}
    }
}

pub fn test_channel_duplex() {
    #[derive(Debug)]
    enum ConnectivityCheck {
        Ping,
        Pong,
        Pang,
    }
    use ConnectivityCheck::*;
    fn main() {
        let n_messages = 3;
        let (requests_tx, requests_rx) = crossbeam::channel::unbounded();
        let (responses_tx, responses_rx) = crossbeam::channel::unbounded();
        std::thread::spawn(move || loop {
            match requests_rx.recv().unwrap() {
                Pong => eprintln!("unexpected pong response"),
                Ping => responses_tx.send(Pong).unwrap(),
                Pang => return,
            }
        });
        for _ in 0..n_messages {
            requests_tx.send(Ping).unwrap();
        }
        for _ in 0..n_messages {
            crossbeam::select! {
            recv(responses_rx) -> msg => println!("{:?}", msg),
            }
        }
        requests_tx.send(Pang).unwrap();
    }
}

pub fn test_svg_singlethread() {
    use std::env;

    use svg::node::element::path::{Command, Data, Position};
    use svg::node::element::{Path, Rectangle};
    use svg::Document;

    use Operation::{
        // <1>
        Forward,   // <1>
        Home,      // <1>
        Noop,      // <1>
        TurnLeft,  // <1>
        TurnRight, // <1>
    }; // <1>
    use Orientation::{
        // <1>
        East,  // <1>
        North, // <1>
        South, // <1>
        West,  // <1>
    }; // <1>

    const WIDTH: isize = 400; // <2>
    const HEIGHT: isize = WIDTH; // <2>

    const HOME_Y: isize = HEIGHT / 2; // <3>
    const HOME_X: isize = WIDTH / 2; // <3>

    const STROKE_WIDTH: usize = 5; // <4>

    #[derive(Debug, Clone, Copy)]
    enum Orientation {
        North, // <5>
        East,  // <5>
        West,  // <5>
        South, // <5>
    }

    #[derive(Debug, Clone, Copy)]
    enum Operation {
        // <6>
        Forward(isize), // <7>
        TurnLeft,
        TurnRight,
        Home,
        Noop(u8), // <8>
    }

    #[derive(Debug)]
    struct Artist {
        // <9>
        x: isize,
        y: isize,
        heading: Orientation,
    }

    impl Artist {
        fn new() -> Artist {
            Artist {
                heading: North,
                x: HOME_X,
                y: HOME_Y,
            }
        }

        fn home(&mut self) {
            self.x = HOME_X;
            self.y = HOME_Y;
        }

        fn forward(&mut self, distance: isize) {
            // <10>
            match self.heading {
                North => self.y += distance,
                South => self.y -= distance,
                West => self.x += distance,
                East => self.x -= distance,
            }
        }

        fn turn_right(&mut self) {
            // <10>
            self.heading = match self.heading {
                North => East,
                South => West,
                West => North,
                East => South,
            }
        }

        fn turn_left(&mut self) {
            // <10>
            self.heading = match self.heading {
                North => West,
                South => East,
                West => South,
                East => North,
            }
        }

        fn wrap(&mut self) {
            // <11>
            if self.x < 0 {
                self.x = HOME_X;
                self.heading = West;
            } else if self.x > WIDTH {
                self.x = HOME_X;
                self.heading = East;
            }

            if self.y < 0 {
                self.y = HOME_Y;
                self.heading = North;
            } else if self.y > HEIGHT {
                self.y = HOME_Y;
                self.heading = South;
            }
        }
    }

    fn parse(input: &str) -> Vec<Operation> {
        use rayon::prelude::*;
        input
            // .bytes() // iter
            .as_bytes() // slice
            .par_iter() // parallel iter
            .map(|&byte| match byte {
                b'0' => Home,
                b'1'..=b'9' => {
                    let distance = (byte - 0x30) as isize;
                    Forward(distance * (HEIGHT / 10))
                }
                b'l' | b'L' => TurnLeft,
                b'r' | b'R' => TurnRight,
                _ => Noop(byte),
            })
            .collect()
        // let mut steps = Vec::<Operation>::new();
        // for byte in input.bytes() {
        //     let step = match byte {
        //         b'0' => Home,
        //         b'1'..=b'9' => {
        //             let distance = (byte - 0x30) as isize; // <12>
        //             Forward(distance * (HEIGHT / 10))
        //         }
        //         b'l' | b'L' => TurnLeft,
        //         b'r' | b'R' => TurnRight,
        //         _ => Noop(byte), // <13>
        //     };
        //     steps.push(step);
        // }
        // steps
    }

    fn convert(operations: &Vec<Operation>) -> Vec<Command> {
        let mut turtle = Artist::new();

        let mut path_data = Vec::<Command>::with_capacity(operations.len());
        let start_at_home = Command::Move(Position::Absolute, (HOME_X, HOME_Y).into());
        path_data.push(start_at_home);

        for op in operations {
            match *op {
                Forward(distance) => turtle.forward(distance),
                TurnLeft => turtle.turn_left(),
                TurnRight => turtle.turn_right(),
                Home => turtle.home(),
                Noop(byte) => {
                    eprintln!("warning: illegal byte encountered: {:?}", byte);
                }
            };

            let path_segment = Command::Line(Position::Absolute, (turtle.x, turtle.y).into());
            path_data.push(path_segment);

            turtle.wrap();
        }
        path_data
    }

    fn generate_svg(path_data: Vec<Command>) -> Document {
        let background = Rectangle::new()
            .set("x", 0)
            .set("y", 0)
            .set("width", WIDTH)
            .set("height", HEIGHT)
            .set("fill", "#ffffff");

        let border = background
            .clone()
            .set("fill-opacity", "0.0")
            .set("stroke", "#cccccc")
            .set("stroke-width", 3 * STROKE_WIDTH);

        let sketch = Path::new()
            .set("fill", "none")
            .set("stroke", "#2f2f2f")
            .set("stroke-width", STROKE_WIDTH)
            .set("stroke-opacity", "0.9")
            .set("d", Data::from(path_data));

        let document = Document::new()
            .set("viewBox", (0, 0, HEIGHT, WIDTH))
            .set("height", HEIGHT)
            .set("width", WIDTH)
            .set("style", "style=\"outline: 5px solid #800000;\"")
            .add(background)
            .add(sketch)
            .add(border);

        document
    }

    fn main() {
        let args = env::args().collect::<Vec<String>>();
        let input = args.get(1).expect("please input svg command");
        let save_to = args.get(2).expect("please input svg file name");
        let operations = parse(input);
        let path_data = convert(&operations);
        let document = generate_svg(path_data);
        svg::save(save_to, &document).unwrap();
    }
    main();
}

pub fn test_svg_multithread() {
    use std::env;

    use crossbeam::channel::unbounded;
    use svg::node::element::path::{Command, Data, Position};
    use svg::node::element::{Path, Rectangle};
    use svg::Document;

    const WIDTH: isize = 400;
    const HEIGHT: isize = WIDTH;

    const HOME_Y: isize = HEIGHT / 2;
    const HOME_X: isize = WIDTH / 2;

    const STROKE_WIDTH: usize = 5;

    #[derive(Debug, Clone, Copy)]
    enum Orientation {
        North,
        East,
        West,
        South,
    }

    #[derive(Debug, Clone, Copy)]
    enum Operation {
        Forward(isize),
        TurnLeft,
        TurnRight,
        Home,
        Noop(u8),
    }

    use Operation::*;
    use Orientation::*;

    #[derive(Debug)]
    struct Artist {
        x: isize,
        y: isize,
        heading: Orientation,
    }

    impl Artist {
        fn new() -> Artist {
            Artist {
                heading: North,
                x: HOME_X,
                y: HOME_Y,
            }
        }

        fn home(&mut self) {
            self.x = HOME_X;
            self.y = HOME_Y;
        }

        fn forward(&mut self, distance: isize) {
            match self.heading {
                North => self.y += distance,
                South => self.y -= distance,
                West => self.x += distance,
                East => self.x -= distance,
            }
        }

        fn turn_right(&mut self) {
            self.heading = match self.heading {
                North => East,
                South => West,
                West => North,
                East => South,
            }
        }

        fn turn_left(&mut self) {
            self.heading = match self.heading {
                North => West,
                South => East,
                West => South,
                East => North,
            }
        }

        fn wrap(&mut self) {
            if self.x < 0 {
                self.x = HOME_X;
                self.heading = West;
            } else if self.x > WIDTH {
                self.x = HOME_X;
                self.heading = East;
            }

            if self.y < 0 {
                self.y = HOME_Y;
                self.heading = North;
            } else if self.y > HEIGHT {
                self.y = HOME_Y;
                self.heading = South;
            }
        }
    }

    enum Work {
        // <1>
        Task((usize, u8)), // <2>
        Finished,          // <3>
    }

    fn parse_byte(byte: u8) -> Operation {
        // <4>
        match byte {
            b'0' => Home,
            b'1'..=b'9' => {
                let distance = (byte - 0x30) as isize;
                Forward(distance * 10)
            }
            b'l' | b'L' => TurnLeft,
            b'r' | b'R' => TurnRight,
            _ => Noop(byte),
        }
    }

    fn parse(input: &str) -> Vec<Operation> {
        let n_threads = 2;
        let (todo_tx, todo_rx) = unbounded(); // <5>
        let (results_tx, results_rx) = unbounded(); // <6>
        let mut n_bytes = 0;
        for (i, byte) in input.bytes().enumerate() {
            todo_tx.send(Work::Task((i, byte))).unwrap(); // <7>
            n_bytes += 1; // <8>
        }

        for _ in 0..n_threads {
            // <9>
            todo_tx.send(Work::Finished).unwrap(); // <9>
        } // <9>

        for _ in 0..n_threads {
            let todo = todo_rx.clone(); // <10>
            let results = results_tx.clone(); // <10>
            std::thread::spawn(move || loop {
                let task = todo.recv();
                let result = match task {
                    Err(_) => break,
                    Ok(Work::Finished) => break,
                    Ok(Work::Task((i, byte))) => (i, parse_byte(byte)),
                };
                results.send(result).unwrap();
            });
        }
        let mut ops = vec![Noop(0); n_bytes]; // <11>
        for _ in 0..n_bytes {
            let (i, op) = results_rx.recv().unwrap();
            ops[i] = op;
        }
        ops
    }

    fn convert(operations: &Vec<Operation>) -> Vec<Command> {
        let mut turtle = Artist::new();

        let mut path_data = Vec::<Command>::with_capacity(operations.len());
        let start_at_home = Command::Move(Position::Absolute, (HOME_X, HOME_Y).into());
        path_data.push(start_at_home);

        for op in operations {
            match *op {
                Forward(distance) => turtle.forward(distance),
                TurnLeft => turtle.turn_left(),
                TurnRight => turtle.turn_right(),
                Home => turtle.home(),
                Noop(byte) => {
                    eprintln!("warning: illegal byte encountered: {:?}", byte);
                }
            };

            let path_segment = Command::Line(Position::Absolute, (turtle.x, turtle.y).into());
            path_data.push(path_segment);

            turtle.wrap();
        }
        path_data
    }

    fn generate_svg(path_data: Vec<Command>) -> Document {
        let background = Rectangle::new()
            .set("x", 0)
            .set("y", 0)
            .set("width", WIDTH)
            .set("height", HEIGHT)
            .set("fill", "#ffffff");

        // let border = background
        //     .clone()
        //     .set("fill-opacity", "0.0")
        //     .set("stroke", "#cccccc")
        //     .set("stroke-width", 3 * STROKE_WIDTH);

        let sketch = Path::new()
            .set("fill", "none")
            .set("stroke", "#2f2f2f")
            .set("stroke-width", STROKE_WIDTH)
            .set("stroke-opacity", "0.9")
            .set("d", Data::from(path_data));

        let document = Document::new()
            .set("viewBox", (0, 0, HEIGHT, WIDTH))
            .set("height", HEIGHT)
            .set("width", WIDTH)
            .set("style", "style=\"outline: 5px solid #800000;\"")
            .add(background)
            .add(sketch);
            // .add(border);

        document
    }

    fn main() {
        let args = env::args().collect::<Vec<String>>();
        let input = args.get(1).expect("please input svg command");
        let save_to = args.get(2).expect("please input svg file name");
        let operations = parse(input);
        let path_data = convert(&operations);
        let document = generate_svg(path_data);
        svg::save(save_to, &document).unwrap();
    }
    main();
}
