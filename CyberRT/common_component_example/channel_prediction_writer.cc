#include "examples.pb.h"

#include "cyber/cyber.h"
#include "cyber/time/rate.h"
#include "cyber/time/time.h"

using apollo::cyber::Rate;
using apollo::cyber::Time;
using testproto::Driver;

int main(int argc, char *argv[]) {
  // init cyber framework
  apollo::cyber::Init(argv[0]);
  // create talker node
  auto talker_node = apollo::cyber::CreateNode("prediction_writer");
  // create talker
  auto talker = talker_node->CreateWriter<Driver>("/apollo/prediction");
  Rate rate(3.0);

  std::string content("apollo_prediction");
  while (apollo::cyber::OK()) {
    static uint64_t seq = 0;
    auto msg = std::make_shared<Driver>();
    msg->set_timestamp(Time::Now().ToNanosecond());
    msg->set_msg_id(seq++);
    msg->set_content(content + std::to_string(seq - 1));
    talker->Write(msg);
    AINFO << "/apollo/prediction sent message, seq=" << (seq - 1) << ";";
    rate.Sleep();
  }
  return 0;
}
