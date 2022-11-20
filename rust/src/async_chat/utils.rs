#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub enum FromClient {
    Join {
        group_name: std::sync::Arc<String>,
    },
    Post {
        group_name: std::sync::Arc<String>,
        message: std::sync::Arc<String>,
    },
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub enum FromServer {
    Message {
        group_name: std::sync::Arc<String>,
        message: std::sync::Arc<String>,
    },
    Error(String),
}

#[test]
fn test_fromclient_json() {
    use std::sync::Arc;
    let from_client = FromClient::Post {
        group_name: Arc::new("Dogs".to_string()),
        message: Arc::new("Samoyeds rock!".to_string()),
    };
    let json = serde_json::to_string(&from_client).unwrap();
    assert_eq!(
        json,
        r#"{"Post":{"group_name":"Dogs","message":"Samoyeds rock!"}}"#
    );
    // assert_eq!(
    //     serde_json::from_str::<FromClient>(&json).unwrap(),
    //     from_client
    // );
}

pub struct Group(
    std::sync::Arc<String>,
    tokio::sync::broadcast::Sender<std::sync::Arc<String>>,
);
impl Group {
    pub fn new(name: std::sync::Arc<String>) -> Group {
        let (sender, receiver) = tokio::sync::broadcast::channel(1000);
        Group(name, sender)
    }
    pub fn join(&self, outbound: std::sync::Arc<Outbound>) {
        async_std::task::spawn(Group::handle_subscriber(
            self.0.clone(),
            self.1.subscribe(),
            outbound,
        ));
    }
    pub fn post(&self, message: std::sync::Arc<String>) {
        let _ignored = self.1.send(message);
    }
    async fn handle_subscriber(
        group_name: std::sync::Arc<String>,
        mut receiver: tokio::sync::broadcast::Receiver<std::sync::Arc<String>>,
        outbound: std::sync::Arc<Outbound>,
    ) {
        loop {
            let packet = match receiver.recv().await {
                Ok(message) => FromServer::Message {
                    group_name: group_name.clone(),
                    message: message.clone(),
                },
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                Err(e) => {
                    FromServer::Error(format!("group: {} error: {}", group_name, e.to_string()))
                }
            };
            if let Err(e) = outbound.send(packet).await {
                println!("send message to client error: {}", e.to_string());
                break;
            }
        }
    }
}

// std::sync::Mutex is just work for multi thread, well async_std::sync::Mutex can work for multi tasks in one/multi thread(s), and task switch may not cause thread switch and may cause same thread lock mutex twice, so async_std::task must use async_std::sync::Mutex.
// std::sync::Mutex will pause the invoke thread when waiting for mutex locked, well async_std::sync::Mutex will switch task when waiting for mutex locked.
// std::sync::Mutex is not Send and so cannot be transfered between threads, and thus std::sync::Mutex can be used in async_std::task::spawn_local but not in async_std::task::spawn (run task in thread pool), well async_std::sync::Mutex is Send
pub struct Outbound(async_std::sync::Mutex<async_std::net::TcpStream>);
impl Outbound {
    pub fn new(to_client: async_std::net::TcpStream) -> Outbound {
        Outbound(async_std::sync::Mutex::new(to_client))
    }
    pub async fn send(&self, packet: FromServer) -> super::ChatResult<()> {
        let mut guard = self.0.lock().await;
        send_as_json(&mut *guard, &packet).await?;
        Ok(())
    }
}

pub async fn send_as_json<S, P>(outbound: &mut S, packet: &P) -> super::ChatResult<()>
where
    S: async_std::io::Write + Unpin,
    P: serde::Serialize,
{
    let mut json = serde_json::to_string(&packet)?;
    json.push('\n');
    use async_std::prelude::*;
    outbound.write_all(json.as_bytes()).await?;
    outbound.flush().await?;
    Ok(())
}

pub fn receive_as_json<S, P>(
    inbound: S,
) -> impl async_std::stream::Stream<Item = super::ChatResult<P>>
where
    S: async_std::io::BufRead + Unpin,
    P: for<'a> serde::de::Deserialize<'a>,
{
    use async_std::prelude::*;
    inbound.lines().map(|line_result| -> super::ChatResult<P> {
        let line = line_result?;
        let parsed = serde_json::from_str::<P>(&line)?;
        Ok(parsed)
    })
}
