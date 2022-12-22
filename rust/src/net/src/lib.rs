pub fn test_http_reqwest() -> Result<(), Box<dyn std::error::Error>> {
    let url = "http://www.rustinaction.com/";
    let mut response = reqwest::get(url)?;
    let content = response.text()?;
    print!("{}", content);
    Ok(())
}

pub fn test_http_tcp() -> std::io::Result<()> {
    use std::io::prelude::*;
    let mut connection = std::net::TcpStream::connect("www.rustinaction.com:80")?;
    // We need to specify the port (80) explicitly,
    // TcpStream does not know that this will become a
    // HTTP request.
    connection.write_all(b"GET / HTTP/1.0")?;
    // GET is the HTTP method, / is the resource we're
    // attempting to access and HTTP/1.0 is the protocol
    // version we're requesting. Why 1.0? It does not
    // support "keep alive" requests, which will allow
    // our stream to close without difficulty.
    connection.write_all(b"\r\n")?;
    // In many networking protocols, \r\n is how a new
    // lines
    connection.write_all(b"Host: www.rustinaction.com")?;
    // The hostname provided on line 5 is actually
    // discarded once it is converted to an IP address.
    // The Host HTTP header allows the server to know
    // which host we're connecting to..
    connection.write_all(b"\r\n\r\n")?;
    // Two blank lines signifies that we've finished the
    // request.
    std::io::copy(&mut connection, &mut std::io::stdout())?;
    // std::io::copy() streams bytes from a Reader to a Writer.
    Ok(())
}

pub fn test_dns() {
    // use trust_dns::op::{Message, MessageType, OpCode, Query};
    // use trust_dns::rr::domain::Name;
    // use trust_dns::rr::record_type::RecordType;
    // use trust_dns::serialize::binary::*;

    // use clap::{arg, command, value_parser, Arg};
    // let matches = command!()
    //     .arg(
    //         arg!([domain_name])
    //             .value_parser(value_parser!(String))
    //             .required(true),
    //     )
    //     .arg(
    //         Arg::new("dns_server")
    //             .short('s')
    //             .long("dns_server")
    //             .default_value("1.1.1.1"),
    //         // arg!([dns_server]).value_parser(value_parser!(String)).default_value("1.1.1.1"),
    //     )
    //     .get_matches();
    // let domain_name_raw = matches
    //     .get_one::<String>("domain_name")
    //     .expect("cannot get domain_name");
    // dbg!(domain_name_raw);
    // let dns_server_raw = matches
    //     .get_one::<String>("dns_server")
    //     .expect("cannot get dns_server");
    // dbg!(dns_server_raw);

    // let domain_name = Name::from_ascii(&domain_name_raw).unwrap();
    // let dns_server: std::net::SocketAddr = format!("{}:53", dns_server_raw)
    //     .parse()
    //     .expect("invalid address");
    // println!("dns_server {:?}", dns_server);

    // let mut request_as_bytes: Vec<u8> = Vec::with_capacity(512); // size is 0
    // let mut response_as_bytes: Vec<u8> = vec![0; 512]; // size is 512

    // let mut msg = Message::new();
    // msg.set_id(rand::random::<u16>())
    //     .set_message_type(MessageType::Query)
    //     .add_query(Query::query(domain_name, RecordType::A))
    //     .set_op_code(OpCode::Query)
    //     .set_recursion_desired(true);

    // let mut encoder = BinEncoder::new(&mut request_as_bytes);
    // msg.emit(&mut encoder).unwrap();

    // let localhost = std::net::UdpSocket::bind("0.0.0.0:0").expect("cannot bind to local socket");
    // let timeout = std::time::Duration::from_secs(3);
    // localhost.set_read_timeout(Some(timeout)).unwrap();
    // localhost.set_nonblocking(false).unwrap();
    // let _amt = localhost
    //     .send_to(&request_as_bytes, dns_server)
    //     .expect("socket misconfigured");
    // let (_amt, _remote) = localhost
    //     .recv_from(&mut response_as_bytes)
    //     .expect("timeout reached");

    // let dns_message = Message::from_vec(&response_as_bytes).expect("unable to parse response");

    // for answer in dns_message.answers() {
    //     if answer.record_type() == RecordType::A {
    //         let resource = answer.rdata();
    //         let ip = resource.to_ip_addr().expect("invalid IP address received");
    //         println!("{}", ip.to_string());
    //     }
    // }
}

pub fn test_MAC() {
    #[derive(Debug)]
    struct MacAddress([u8; 6]);
    impl std::fmt::Display for MacAddress {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let octet = &self.0;
            write!(
                f,
                "{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
                octet[0], octet[1], octet[2], octet[3], octet[4], octet[5]
            )
        }
    }
    impl MacAddress {
        fn new() -> MacAddress {
            let mut octets: [u8; 6] = [0; 6];
            use rand::RngCore;
            rand::thread_rng().fill_bytes(&mut octets);
            octets[0] |= 0b_0000_0011;
            MacAddress { 0: octets }
        }
        fn is_local(&self) -> bool {
            (self.0[0] & 0b_0000_0010) == 0b_0000_0010
        }
        fn is_unicast(&self) -> bool {
            (self.0[0] & 0b_0000_0001) == 0b_0000_0001
        }
    }
    let mac = MacAddress::new();
    assert!(mac.is_local());
    assert!(mac.is_unicast());
    println!("mac: {}", mac);
}

mod dns {
    use std::error::Error;
    use std::net::{SocketAddr, UdpSocket};
    use std::time::Duration;

    use trust_dns::op::{Message, MessageType, OpCode, Query};
    use trust_dns::proto::error::ProtoError;
    use trust_dns::rr::domain::Name;
    use trust_dns::rr::record_type::RecordType;
    use trust_dns::serialize::binary::*;

    fn message_id() -> u16 {
        let candidate = rand::random();
        if candidate == 0 {
            return message_id();
        }
        candidate
    }

    #[derive(Debug)]
    pub enum DnsError {
        ParseDomainName(ProtoError),
        ParseDnsServerAddress(std::net::AddrParseError),
        Encoding(ProtoError),
        Decoding(ProtoError),
        Network(std::io::Error),
        Sending(std::io::Error),
        Receving(std::io::Error),
    }

    impl std::fmt::Display for DnsError {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "{:#?}", self)
        }
    }

    impl std::error::Error for DnsError {}

    pub fn resolve(
        dns_server_address: &str,
        domain_name: &str,
    ) -> Result<Option<std::net::IpAddr>, Box<dyn Error>> {
        let domain_name = Name::from_ascii(domain_name).map_err(DnsError::ParseDomainName)?;

        let dns_server_address = format!("{}:53", dns_server_address);
        let dns_server: SocketAddr = dns_server_address
            .parse()
            .map_err(DnsError::ParseDnsServerAddress)?;

        let mut request_buffer: Vec<u8> = Vec::with_capacity(64);
        let mut response_buffer: Vec<u8> = vec![0; 512];

        let mut request = Message::new();
        request.add_query(Query::query(domain_name, RecordType::A));

        request
            .set_id(message_id())
            .set_message_type(MessageType::Query)
            .set_op_code(OpCode::Query)
            .set_recursion_desired(true);

        let localhost = UdpSocket::bind("0.0.0.0:0").map_err(DnsError::Network)?;

        let timeout = Duration::from_secs(5);
        localhost
            .set_read_timeout(Some(timeout))
            .map_err(DnsError::Network)?;

        localhost
            .set_nonblocking(false)
            .map_err(DnsError::Network)?;

        let mut encoder = BinEncoder::new(&mut request_buffer);
        request.emit(&mut encoder).map_err(DnsError::Encoding)?;

        let _n_bytes_sent = localhost
            .send_to(&request_buffer, dns_server)
            .map_err(DnsError::Sending)?;

        loop {
            // <8>
            let (_b_bytes_recv, remote_port) = localhost
                .recv_from(&mut response_buffer)
                .map_err(DnsError::Receving)?;

            if remote_port == dns_server {
                break;
            }
        }

        let response = Message::from_vec(&response_buffer).map_err(DnsError::Decoding)?;

        for answer in response.answers() {
            if answer.record_type() == RecordType::A {
                let resource = answer.rdata();
                let server_ip = resource.to_ip_addr().expect("invalid IP address received");
                return Ok(Some(server_ip));
            }
        }

        Ok(None)
    }
}

mod ethernet {
    use rand;
    use std::fmt;
    use std::fmt::Display;

    use rand::RngCore;
    use smoltcp::wire;

    #[derive(Debug)]
    pub struct MacAddress([u8; 6]);

    impl Display for MacAddress {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let octet = self.0;
            write!(
                f,
                "{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
                octet[0], octet[1], octet[2], octet[3], octet[4], octet[5]
            )
        }
    }

    impl MacAddress {
        pub fn new() -> MacAddress {
            let mut octets: [u8; 6] = [0; 6];
            rand::thread_rng().fill_bytes(&mut octets); // <1>
            octets[0] |= 0b_0000_0010; // <2>
            octets[0] &= 0b_1111_1110; // <3>
            MacAddress { 0: octets }
        }
    }

    impl Into<wire::EthernetAddress> for MacAddress {
        fn into(self) -> wire::EthernetAddress {
            wire::EthernetAddress { 0: self.0 }
        }
    }
}

mod http {
    use std::collections::BTreeMap;
    use std::fmt;
    use std::net::IpAddr;
    use std::os::unix::io::AsRawFd;

    use smoltcp::iface::{EthernetInterfaceBuilder, NeighborCache, Routes};
    use smoltcp::phy::{wait as phy_wait, TapInterface};
    use smoltcp::socket::{SocketSet, TcpSocket, TcpSocketBuffer};
    use smoltcp::time::Instant;
    use smoltcp::wire::{EthernetAddress, IpAddress, IpCidr, Ipv4Address};
    use url::Url;

    #[derive(Debug)]
    enum HttpState {
        Connect,
        Request,
        Response,
    }

    #[derive(Debug)]
    pub enum UpstreamError {
        Network(smoltcp::Error),
        InvalidUrl,
        Content(std::str::Utf8Error),
    }

    impl fmt::Display for UpstreamError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{:?}", self)
        }
    }

    impl From<smoltcp::Error> for UpstreamError {
        fn from(error: smoltcp::Error) -> Self {
            UpstreamError::Network(error)
        }
    }

    impl From<std::str::Utf8Error> for UpstreamError {
        fn from(error: std::str::Utf8Error) -> Self {
            UpstreamError::Content(error)
        }
    }

    fn random_port() -> u16 {
        49152 + rand::random::<u16>() % 16384
    }

    pub fn get(
        tap: TapInterface,
        mac: EthernetAddress,
        addr: IpAddr,
        url: Url,
    ) -> Result<(), UpstreamError> {
        let domain_name = url.host_str().ok_or(UpstreamError::InvalidUrl)?;

        let neighbor_cache = NeighborCache::new(BTreeMap::new());

        let tcp_rx_buffer = TcpSocketBuffer::new(vec![0; 1024]);
        let tcp_tx_buffer = TcpSocketBuffer::new(vec![0; 1024]);
        let tcp_socket = TcpSocket::new(tcp_rx_buffer, tcp_tx_buffer);

        let ip_addrs = [IpCidr::new(IpAddress::v4(192, 168, 42, 1), 24)];

        let fd = tap.as_raw_fd();
        let mut routes = Routes::new(BTreeMap::new());
        let default_gateway = Ipv4Address::new(192, 168, 42, 100);
        routes.add_default_ipv4_route(default_gateway).unwrap();
        let mut iface = EthernetInterfaceBuilder::new(tap)
            .ethernet_addr(mac)
            .neighbor_cache(neighbor_cache)
            .ip_addrs(ip_addrs)
            .routes(routes)
            .finalize();

        let mut sockets = SocketSet::new(vec![]);
        let tcp_handle = sockets.add(tcp_socket);

        let http_header = format!(
            "GET {} HTTP/1.0\r\nHost: {}\r\nConnection: close\r\n\r\n",
            url.path(),
            domain_name,
        );

        let mut state = HttpState::Connect;
        'http: loop {
            let timestamp = Instant::now();
            match iface.poll(&mut sockets, timestamp) {
                Ok(_) => {}
                Err(smoltcp::Error::Unrecognized) => {}
                Err(e) => {
                    eprintln!("error: {:?}", e);
                }
            }

            {
                let mut socket = sockets.get::<TcpSocket>(tcp_handle);

                state = match state {
                    HttpState::Connect if !socket.is_active() => {
                        eprintln!("connecting");
                        socket.connect((addr, 80), random_port())?;
                        HttpState::Request
                    }

                    HttpState::Request if socket.may_send() => {
                        eprintln!("sending request");
                        socket.send_slice(http_header.as_ref())?;
                        HttpState::Response
                    }

                    HttpState::Response if socket.can_recv() => {
                        eprintln!("can_recv");
                        socket.recv(|raw_data| {
                            let output = String::from_utf8_lossy(raw_data);
                            // println!("{}", output);
                            (raw_data.len(), ())
                        })?;
                        HttpState::Response
                    }

                    HttpState::Response if !socket.may_recv() => {
                        eprintln!("received complete response");
                        break 'http;
                    }
                    _ => state,
                }
            }

            phy_wait(fd, iface.poll_delay(&sockets, timestamp)).expect("wait error");
        }
        eprintln!("Ok");
        Ok(())
    }
}

pub fn test_mget() {
    // std::env::args().for_each(|x| {println!("{}", x)});
    use clap::{App, Arg};
    use smoltcp::phy::TapInterface;
    use url::Url;

    let app = App::new("mget")
        .about("GET a webpage, manually")
        .arg(Arg::with_name("url").required(true))
        .arg(Arg::with_name("tap-device").required(true))
        .arg(Arg::with_name("dns-server").default_value("1.1.1.1"))
        .get_matches();

    let url_text = app.value_of("url").unwrap();
    let dns_server_text = app.value_of("dns-server").unwrap();
    let tap_text = app.value_of("tap-device").unwrap();

    let url = Url::parse(url_text).expect("error: unable to parse <url> as a URL");

    if url.scheme() != "http" {
        eprintln!("error: only HTTP protocol supported");
        return;
    }

    let tap = TapInterface::new(&tap_text).expect(
        "error: unable to use <tap-device> as a \
           network interface",
    );

    let domain_name = url.host_str().expect("domain name required");

    let _dns_server: std::net::Ipv4Addr = dns_server_text.parse().expect(
        "error: unable to parse <dns-server> as an \
             IPv4 address",
    );

    let addr = dns::resolve(dns_server_text, domain_name).unwrap().unwrap();

    let mac = ethernet::MacAddress::new().into();

    http::get(tap, mac, addr, url).unwrap();
}
