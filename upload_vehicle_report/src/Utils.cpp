#include "Utils.h"
#include "ShareResource.h"
#include <fstream>

using namespace v2x;

thread_local std::string ThreadUtil::current_thread_name = "v2x";

Config::Config()
{
	std::ifstream cfgYaml(ShareResource::configFilePath);
	if (!cfgYaml.good())
	{
		PRINT_ERROR("config file cfg.yaml not found");
		exit(ERROR_CONFIG);
	}
	cfgYaml.close();
	cfgNode = YAML::LoadFile("cfg.yaml");
	if (!cfgNode)
	{
		PRINT_ERROR("load cfg.yaml failed");
		exit(ERROR_CONFIG);
	}
}

YAML::Node Config::getYAMLConfigItem(const std::string &nodeName)
{
	if (nodeName.empty())
	{
		PRINT_ERROR("empty config item");
		exit(ERROR_CONFIG_ITEM);
	}
	YAML::Node getNode = cfgNode[nodeName];
	if (!getNode)
	{
		PRINT_ERROR("%s is not valid config item", nodeName.c_str());
		exit(ERROR_CONFIG_ITEM);
	}
	return std::move(getNode);
}
