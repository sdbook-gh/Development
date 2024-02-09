#include "HelloWorldPubSubTypes.h"
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/publisher/DataWriter.hpp>
#include <fastdds/dds/publisher/DataWriterListener.hpp>
#include <fastdds/dds/publisher/Publisher.hpp>
#include <fastdds/dds/topic/TypeSupport.hpp>
#include <fastdds/rtps/transport/UDPv4TransportDescriptor.h>
#include <fastdds/rtps/transport/shared_mem/SharedMemTransportDescriptor.h>

using namespace eprosima::fastdds::dds;
using namespace eprosima::fastdds::rtps;
using namespace eprosima::fastrtps::rtps;
class HelloWorldPublisher {
private:
  DomainParticipant *participant_;
  Publisher *publisher_;
  Topic *topic_;
  DataWriter *writer_;
  TypeSupport type_;
  class PubListener : public DataWriterListener {
  public:
    PubListener() : matched_(0) {}
    ~PubListener() override {}
    void on_publication_matched(DataWriter *, const PublicationMatchedStatus &info) override {
      if (info.current_count_change == 1) {
        matched_ = info.total_count;
        std::cout << "Publisher matched." << std::endl;
      } else if (info.current_count_change == -1) {
        matched_ = info.total_count;
        std::cout << "Publisher unmatched." << std::endl;
      } else {
        std::cout << info.current_count_change
                  << " is not a valid value for PublicationMatchedStatus current count change." << std::endl;
      }
    }
    std::atomic_int matched_;
  } listener_;

public:
  HelloWorldPublisher()
    : participant_(nullptr),
      publisher_(nullptr),
      topic_(nullptr),
      writer_(nullptr),
      type_(new HelloWorldPubSubType()) {}
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
  //! Initialize the publisher
  bool init() {
    DomainParticipantQos participantQos;
    // participantQos.wire_protocol().builtin.discovery_config.discoveryProtocol = DiscoveryProtocol_t::SIMPLE;
    // participantQos.wire_protocol().builtin.discovery_config.use_SIMPLE_EndpointDiscoveryProtocol = true;
    // participantQos.wire_protocol().builtin.discovery_config.m_simpleEDP.use_PublicationReaderANDSubscriptionWriter =
    //   true;
    // participantQos.wire_protocol().builtin.discovery_config.m_simpleEDP.use_PublicationWriterANDSubscriptionReader =
    //   true;
    // participantQos.wire_protocol().builtin.discovery_config.leaseDuration = eprosima::fastrtps::c_TimeInfinite;
    // Explicit configuration of SharedMem transport
    participantQos.transport().use_builtin_transports = false;
    auto shm_transport = std::make_shared<SharedMemTransportDescriptor>();
    shm_transport->segment_size(2 * 1024 * 1024);
    participantQos.transport().user_transports.push_back(shm_transport);
    participantQos.name("Participant_publisher");
    participant_ = DomainParticipantFactory::get_instance()->create_participant(0, participantQos);
    if (participant_ == nullptr) {
      return false;
    }
    // Register the Type
    type_.register_type(participant_);
    // Create the publications Topic
    topic_ = participant_->create_topic("HelloWorldTopic", "HelloWorld", TOPIC_QOS_DEFAULT);
    if (topic_ == nullptr) {
      return false;
    }
    // Create the Publisher
    publisher_ = participant_->create_publisher(PUBLISHER_QOS_DEFAULT, nullptr);
    if (publisher_ == nullptr) {
      return false;
    }
    // Create the DataWriter
    writer_ = publisher_->create_datawriter(topic_, DATAWRITER_QOS_DEFAULT, &listener_);
    if (writer_ == nullptr) {
      return false;
    }
    return true;
  }
  //! Send a publication
  bool publish(int index) {
    if (listener_.matched_ > 0) {
      void *sample = nullptr;
      if (ReturnCode_t::RETCODE_OK == writer_->loan_sample(sample)) {
        std::cout << "Preparing sample at address " << sample << std::endl;
        HelloWorld *data = static_cast<HelloWorld *>(sample);
        data->index() = index;
        memcpy(data->message().data(), "FastDDS message ", strlen("FastDDS message") + 1);
        writer_->write(sample);
        std::cout << "Message: " << data->message().data() << " with index: " << data->index() << " SENT" << std::endl;
      }
      return true;
    }
    return false;
  }
  //! Run the Publisher
  void run(uint32_t samples) {
    uint32_t samples_sent = 0;
    while (samples_sent < samples) {
      if (publish(samples_sent)) {
        samples_sent++;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
  }
};

int main(int argc, char **argv) {
  std::cout << "Starting publisher." << std::endl;
  uint32_t samples = 10;

  HelloWorldPublisher *mypub = new HelloWorldPublisher();
  if (mypub->init()) {
    mypub->run(samples);
  }

  delete mypub;
  return 0;
}
