#include "HelloWorldPubSubTypes.h"

#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>

#include <fastdds/dds/publisher/DataWriter.hpp>
#include <fastdds/dds/publisher/DataWriterListener.hpp>
#include <fastdds/dds/publisher/Publisher.hpp>
#include <fastdds/dds/publisher/qos/DataWriterQos.hpp>

#include <fastdds/rtps/transport/shared_mem/SharedMemTransportDescriptor.h>

#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/DataReaderListener.hpp>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include <fastdds/dds/subscriber/qos/DataReaderQos.hpp>

#include <chrono>
#include <thread>

using namespace eprosima::fastdds::dds;
using namespace eprosima::fastdds::rtps;
using namespace eprosima::fastrtps::rtps;

class HelloWorldPublisher {
public:
  HelloWorldPublisher()
      : participant_(nullptr), publisher_(nullptr), topic_(nullptr),
        writer_(nullptr), type_(new HelloWorldPubSubType()) {}

  virtual ~HelloWorldPublisher() {
    if (writer_ != nullptr) {
      publisher_->delete_datawriter(writer_);
    }
    if (publisher_ != nullptr) {
      participant_->delete_publisher(publisher_);
    }
    if (topic_ != nullptr) {
      participant_->delete_topic(topic_);
    }
    DomainParticipantFactory::get_instance()->delete_participant(participant_);
  }

  //! Initialize
  bool init() {
    hello_.index(0);
    hello_.message("HelloWorld");

    // CREATE THE PARTICIPANT
    DomainParticipantQos pqos;
    pqos.wire_protocol().builtin.discovery_config.discoveryProtocol =
        DiscoveryProtocol_t::SIMPLE;
    pqos.wire_protocol()
        .builtin.discovery_config.use_SIMPLE_EndpointDiscoveryProtocol = true;
    pqos.wire_protocol()
        .builtin.discovery_config.m_simpleEDP
        .use_PublicationReaderANDSubscriptionWriter = true;
    pqos.wire_protocol()
        .builtin.discovery_config.m_simpleEDP
        .use_PublicationWriterANDSubscriptionReader = true;
    pqos.wire_protocol().builtin.discovery_config.leaseDuration =
        eprosima::fastrtps::c_TimeInfinite;
    pqos.name("Participant_pub");

    // Explicit configuration of SharedMem transport
    pqos.transport().use_builtin_transports = false;

    auto shm_transport = std::make_shared<SharedMemTransportDescriptor>();
    shm_transport->segment_size(2 * 1024 * 1024);
    pqos.transport().user_transports.push_back(shm_transport);

    participant_ =
        DomainParticipantFactory::get_instance()->create_participant(0, pqos);

    if (participant_ == nullptr) {
      return false;
    }

    // REGISTER THE TYPE
    type_.register_type(participant_);

    // CREATE THE PUBLISHER
    publisher_ = participant_->create_publisher(PUBLISHER_QOS_DEFAULT);

    if (publisher_ == nullptr) {
      return false;
    }

    // CREATE THE TOPIC
    topic_ = participant_->create_topic("HelloWorldSharedMemTopic",
                                        "HelloWorld", TOPIC_QOS_DEFAULT);

    if (topic_ == nullptr) {
      return false;
    }

    // CREATE THE DATAWRITER
    DataWriterQos wqos;
    wqos.history().kind = KEEP_LAST_HISTORY_QOS;
    wqos.history().depth = 30;
    wqos.resource_limits().max_samples = 50;
    wqos.resource_limits().allocated_samples = 20;
    wqos.reliable_writer_qos().times.heartbeatPeriod.seconds = 2;
    wqos.reliable_writer_qos().times.heartbeatPeriod.nanosec =
        200 * 1000 * 1000;
    wqos.reliability().kind = RELIABLE_RELIABILITY_QOS;
    wqos.publish_mode().kind = ASYNCHRONOUS_PUBLISH_MODE;

    writer_ = publisher_->create_datawriter(topic_, wqos, &listener_);

    if (writer_ == nullptr) {
      return false;
    }

    return true;
  }

  //! Publish a sample
  bool publish(bool waitForListener = true) {
    if (listener_.firstConnected_ || !waitForListener ||
        listener_.matched_ > 0) {
      hello_.index(hello_.index() + 1);
      writer_->write(&hello_);
      return true;
    }
    return false;
  }

  //! Run for number samples
  void run(uint32_t samples, uint32_t sleep) {
    stop_ = false;
    std::thread thread(&HelloWorldPublisher::runThread, this, samples, sleep);
    if (samples == 0) {
      std::cout << "Publisher running. Please press enter to stop the "
                   "Publisher at any time."
                << std::endl;
      std::cin.ignore();
      stop_ = true;
    } else {
      std::cout << "Publisher running " << samples << " samples." << std::endl;
    }
    thread.join();
  }

private:
  HelloWorld hello_;

  eprosima::fastdds::dds::DomainParticipant *participant_;

  eprosima::fastdds::dds::Publisher *publisher_;

  eprosima::fastdds::dds::Topic *topic_;

  eprosima::fastdds::dds::DataWriter *writer_;

  bool stop_;

  class PubListener : public eprosima::fastdds::dds::DataWriterListener {
  public:
    PubListener() : matched_(0), firstConnected_(false) {}

    ~PubListener() override {}

    void on_publication_matched(
        eprosima::fastdds::dds::DataWriter *,
        const eprosima::fastdds::dds::PublicationMatchedStatus &info) {
      if (info.current_count_change == 1) {
        matched_ = info.total_count;
        firstConnected_ = true;
        std::cout << "Publisher matched." << std::endl;
      } else if (info.current_count_change == -1) {
        matched_ = info.total_count;
        std::cout << "Publisher unmatched." << std::endl;
      } else {
        std::cout << info.current_count_change
                  << " is not a valid value for PublicationMatchedStatus "
                     "current count change"
                  << std::endl;
      }
    }

    int matched_;

    bool firstConnected_;
  } listener_;

  void runThread(uint32_t samples, uint32_t sleep) {
    if (samples == 0) {
      while (!stop_) {
        if (publish(false)) {
          std::cout << "Message: " << hello_.message()
                    << " with index: " << hello_.index() << " SENT"
                    << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep));
      }
    } else {
      for (uint32_t i = 0; i < samples; ++i) {
        if (!publish()) {
          --i;
        } else {
          std::cout << "Message: " << hello_.message()
                    << " with index: " << hello_.index() << " SENT"
                    << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep));
      }
    }
  }

  eprosima::fastdds::dds::TypeSupport type_;
};

class HelloWorldSubscriber {
public:
  HelloWorldSubscriber()
      : participant_(nullptr), subscriber_(nullptr), topic_(nullptr),
        reader_(nullptr), type_(new HelloWorldPubSubType()) {}
  virtual ~HelloWorldSubscriber() {
    if (reader_ != nullptr) {
      subscriber_->delete_datareader(reader_);
    }
    if (topic_ != nullptr) {
      participant_->delete_topic(topic_);
    }
    if (subscriber_ != nullptr) {
      participant_->delete_subscriber(subscriber_);
    }
    DomainParticipantFactory::get_instance()->delete_participant(participant_);
  }

  //! Initialize the subscriber
  bool init() {
    // CREATE THE PARTICIPANT
    DomainParticipantQos pqos;
    pqos.wire_protocol().builtin.discovery_config.discoveryProtocol =
        DiscoveryProtocol_t::SIMPLE;
    pqos.wire_protocol()
        .builtin.discovery_config.use_SIMPLE_EndpointDiscoveryProtocol = true;
    pqos.wire_protocol()
        .builtin.discovery_config.m_simpleEDP
        .use_PublicationReaderANDSubscriptionWriter = true;
    pqos.wire_protocol()
        .builtin.discovery_config.m_simpleEDP
        .use_PublicationWriterANDSubscriptionReader = true;
    pqos.wire_protocol().builtin.discovery_config.leaseDuration =
        eprosima::fastrtps::c_TimeInfinite;
    pqos.name("Participant_sub");

    // Explicit configuration of SharedMem transport
    pqos.transport().use_builtin_transports = false;

    auto sm_transport = std::make_shared<SharedMemTransportDescriptor>();
    sm_transport->segment_size(2 * 1024 * 1024);
    pqos.transport().user_transports.push_back(sm_transport);

    participant_ =
        DomainParticipantFactory::get_instance()->create_participant(0, pqos);

    if (participant_ == nullptr) {
      return false;
    }

    // REGISTER THE TYPE
    type_.register_type(participant_);

    // CREATE THE SUBSCRIBER
    subscriber_ = participant_->create_subscriber(SUBSCRIBER_QOS_DEFAULT);

    if (subscriber_ == nullptr) {
      return false;
    }

    // CREATE THE TOPIC
    topic_ = participant_->create_topic("HelloWorldSharedMemTopic",
                                        "HelloWorld", TOPIC_QOS_DEFAULT);

    if (topic_ == nullptr) {
      return false;
    }

    // CREATE THE DATAREADER
    DataReaderQos rqos;
    rqos.history().kind = KEEP_LAST_HISTORY_QOS;
    rqos.history().depth = 30;
    rqos.resource_limits().max_samples = 50;
    rqos.resource_limits().allocated_samples = 20;
    rqos.reliability().kind = RELIABLE_RELIABILITY_QOS;
    rqos.durability().kind = TRANSIENT_LOCAL_DURABILITY_QOS;

    reader_ = subscriber_->create_datareader(topic_, rqos, &listener_);

    if (reader_ == nullptr) {
      return false;
    }

    return true;
  }

  //! RUN the subscriber
  void run() {
    std::cout << "Subscriber running. Please press enter to stop the Subscriber"
              << std::endl;
    std::cin.ignore();
  }

  //! Run the subscriber until number samples have been received.
  void run(uint32_t number) {
    std::cout << "Subscriber running until " << number
              << "samples have been received" << std::endl;
    while (number > listener_.samples_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
  }

private:
  eprosima::fastdds::dds::DomainParticipant *participant_;

  eprosima::fastdds::dds::Subscriber *subscriber_;

  eprosima::fastdds::dds::Topic *topic_;

  eprosima::fastdds::dds::DataReader *reader_;

  eprosima::fastdds::dds::TypeSupport type_;

  class SubListener : public eprosima::fastdds::dds::DataReaderListener {
  public:
    SubListener() : matched_(0), samples_(0) {}

    ~SubListener() override {}

    void
    on_data_available(eprosima::fastdds::dds::DataReader *reader) override {
      SampleInfo info;
      if (reader->take_next_sample(&hello_, &info) ==
          ReturnCode_t::RETCODE_OK) {
        if (info.instance_state == ALIVE_INSTANCE_STATE) {
          samples_++;
          // Print your structure data here.
          std::cout << "Message " << hello_.message() << " " << hello_.index()
                    << " RECEIVED" << std::endl;
        }
      }
    }
    void on_subscription_matched(
        eprosima::fastdds::dds::DataReader *reader,
        const eprosima::fastdds::dds::SubscriptionMatchedStatus &info)
        override {
      if (info.current_count_change == 1) {
        matched_ = info.total_count;
        std::cout << "Subscriber matched." << std::endl;
      } else if (info.current_count_change == -1) {
        matched_ = info.total_count;
        std::cout << "Subscriber unmatched." << std::endl;
      } else {
        std::cout << info.current_count_change
                  << " is not a valid value for SubscriptionMatchedStatus "
                     "current count change"
                  << std::endl;
      }
    }

    HelloWorld hello_;

    int matched_;

    uint32_t samples_;
  } listener_;
};

#include <fastrtps/Domain.h>
#include <fastrtps/log/Log.h>

using namespace eprosima;
using namespace fastrtps;
using namespace rtps;
int main(int argc, char **argv) {
  std::cout << "Starting " << std::endl;
  int type = 1;
  if (argc > 1) {
    if (strcmp(argv[1], "publisher") == 0) {
      type = 1;
    } else if (strcmp(argv[1], "subscriber") == 0) {
      type = 2;
    }
  } else {
    std::cout << "publisher OR subscriber argument needed" << std::endl;
    Log::Reset();
    return 0;
  }

  switch (type) {
  case 1: {
    HelloWorldPublisher mypub;
    if (mypub.init()) {
      mypub.run(10, 100);
    }
    break;
  }
  case 2: {
    HelloWorldSubscriber mysub;
    if (mysub.init()) {
      mysub.run();
    }
    break;
  }
  }
  Domain::stopAll();
  Log::Reset();
  return 0;
}
