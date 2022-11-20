pub type ChatError = Box<dyn std::error::Error + Send + Sync + 'static>;
pub type ChatResult<T> = Result<T, ChatError>;

mod utils;

pub async fn send_commands(mut to_server: async_std::net::TcpStream) -> ChatResult<()> {
    use async_std::prelude::*;
    println!(
        "Commands:\n\
        join GROUP\n\
        post GROUP MESSAGE...\n\
        Type Control-D (on Unix) or Control-Z (on Windows) \
        to close the connection."
    );
    let mut command_lines = async_std::io::BufReader::new(async_std::io::stdin()).lines();
    while let Some(command_result) = command_lines.next().await {
        let command = command_result?;
        let request = match parse_command(&command) {
            Some(request) => request,
            None => continue,
        };
        match utils::send_as_json(&mut to_server, &request).await {
            Err(e) => {
                println!("send_as_json error: {}", e.to_string())
            }
            _ => (),
        }
    }
    Ok(())
}

pub async fn handle_replies(from_server: async_std::net::TcpStream) -> ChatResult<()> {
    use async_std::prelude::*;
    let message_buffer = async_std::io::BufReader::new(from_server);
    let mut reply_stream = utils::receive_as_json(message_buffer);
    while let Some(reply) = reply_stream.next().await {
        match reply {
            Ok(r) => match r {
                utils::FromServer::Message {
                    group_name,
                    message,
                } => {
                    println!("message posted to {}: {}", group_name, message);
                }
                utils::FromServer::Error(message) => {
                    println!("error from server: {}", message);
                }
            },
            Err(e) => {
                println!("receive_as_json error: {}", e.to_string());
            }
        }
    }
    Ok(())
}

fn parse_command(line: &str) -> Option<utils::FromClient> {
    let (command, rest) = get_next_token(line)?;
    if command == "post" {
        let (group, rest) = get_next_token(rest)?;
        let message = rest.trim_start().to_string();
        return Some(utils::FromClient::Post {
            group_name: std::sync::Arc::new(group.to_string()),
            message: std::sync::Arc::new(message),
        });
    } else if command == "join" {
        let (group, rest) = get_next_token(rest)?;
        if !rest.trim_start().is_empty() {
            return None;
        }
        return Some(utils::FromClient::Join {
            group_name: std::sync::Arc::new(group.to_string()),
        });
    } else {
        eprintln!("Unrecognized command: {:?}", line);
        return None;
    }
}

fn get_next_token(mut input: &str) -> Option<(&str, &str)> {
    input = input.trim_start();
    if input.is_empty() {
        return None;
    }
    match input.find(char::is_whitespace) {
        Some(space) => Some((&input[0..space], &input[space..])),
        None => Some((input, "")),
    }
}

pub struct GroupTable(
    std::sync::Mutex<std::collections::HashMap<std::sync::Arc<String>, std::sync::Arc<utils::Group>>>,
);
impl GroupTable {
    pub fn new() -> GroupTable {
        GroupTable(std::sync::Mutex::new(std::collections::HashMap::new()))
    }
    pub fn get(&self, name: &String) -> Option<std::sync::Arc<utils::Group>> {
        self.0.lock().unwrap().get(name).cloned()
    }
    pub fn get_or_create(&self, name: std::sync::Arc<String>) -> std::sync::Arc<utils::Group> {
        self.0
            .lock()
            .unwrap()
            .entry(name.clone())
            .or_insert_with(|| std::sync::Arc::new(utils::Group::new(name)))
            .clone()
    }
}

pub async fn serve(
    socket: async_std::net::TcpStream,
    group_table: std::sync::Arc<GroupTable>,
) -> ChatResult<()> {
    use async_std::prelude::*;
    let outbound = std::sync::Arc::new(utils::Outbound::new(socket.clone()));
    let buffered = async_std::io::BufReader::new(socket);
    let mut from_client = utils::receive_as_json(buffered);
    while let Some(request_result) = from_client.next().await {
        match request_result {
            Ok(request) => match request {
                utils::FromClient::Join { group_name } => {
                    let group = group_table.get_or_create(group_name);
                    group.join(outbound.clone());
                }
                utils::FromClient::Post {
                    group_name,
                    message,
                } => match group_table.get(&group_name) {
                    Some(group) => {
                        group.post(message);
                    }
                    None => {
                        let message = utils::FromServer::Error(format!(
                            "Group '{}' does not exist",
                            group_name
                        ));
                        outbound.send(message).await?;
                    }
                },
            },
            Err(e) => {
                println!("receive_as_json error {}", e.to_string());
            }
        }
    }
    Ok(())
}
