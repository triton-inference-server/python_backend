// Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "custom_metrics_registry.h"
#include "triton/common/triton_json.h"

namespace triton { namespace backend { namespace python {

PbMetric::PbMetric(
    const std::string& labels, TRITONSERVER_MetricFamily* metric_family)
    : labels_(labels), value_(0.0), metric_(nullptr)
{
  std::vector<const TRITONSERVER_Parameter*> labels_params;
  ParseLabels(labels_params, labels);
  THROW_IF_TRITON_ERROR(TRITONSERVER_MetricNew(
      &metric_, metric_family, labels_params.data(), labels_params.size()));
  for (const auto label : labels_params) {
    TRITONSERVER_ParameterDelete(const_cast<TRITONSERVER_Parameter*>(label));
  }
}

PbMetric::~PbMetric()
{
  if (metric_ != nullptr) {
    TRITONSERVER_MetricDelete(metric_);
  }
}

const std::string&
PbMetric::Labels()
{
  return labels_;
}

void
PbMetric::ParseLabels(
    std::vector<const TRITONSERVER_Parameter*>& labels_params,
    const std::string& labels)
{
  triton::common::TritonJson::Value labels_json;
  THROW_IF_TRITON_ERROR(labels_json.Parse(labels));

  std::vector<std::string> members;
  labels_json.Members(&members);
  for (const auto& member : members) {
    std::string value;
    THROW_IF_TRITON_ERROR(labels_json.MemberAsString(member.c_str(), &value));
    labels_params.emplace_back(TRITONSERVER_ParameterNew(
        member.c_str(), TRITONSERVER_PARAMETER_STRING, value.c_str()));
  }
}

void
PbMetric::Increment(const double& value)
{
  THROW_IF_TRITON_ERROR(TRITONSERVER_MetricIncrement(metric_, value));
  UpdateValue();
}

void
PbMetric::SetValue(const double& value)
{
  THROW_IF_TRITON_ERROR(TRITONSERVER_MetricSet(metric_, value));
  UpdateValue();
}

double
PbMetric::Value()
{
  UpdateValue();
  return value_;
}

void
PbMetric::UpdateValue()
{
  THROW_IF_TRITON_ERROR(TRITONSERVER_MetricValue(metric_, &value_));
}

PbMetricFamily::PbMetricFamily(
    const std::string& name, const std::string& description,
    const MetricKind& kind)
    : name_(name), description_(description), kind_(kind),
      metric_family_(nullptr)
{
  TRITONSERVER_MetricKind triton_kind = ToTritonServerMetricKind(kind);
  THROW_IF_TRITON_ERROR(TRITONSERVER_MetricFamilyNew(
      &metric_family_, triton_kind, name_.c_str(), description_.c_str()));
}

PbMetricFamily::~PbMetricFamily()
{
  std::lock_guard<std::mutex> lock(metric_map_mu_);
  metric_map_.clear();
  if (metric_family_ != nullptr) {
    LOG_IF_ERROR(
        TRITONSERVER_MetricFamilyDelete(metric_family_),
        "deleting metric family");
  }
}

TRITONSERVER_MetricKind
PbMetricFamily::ToTritonServerMetricKind(const MetricKind& kind)
{
  switch (kind) {
    case COUNTER:
      return TRITONSERVER_METRIC_KIND_COUNTER;
    case GAUGE:
      return TRITONSERVER_METRIC_KIND_GAUGE;
    default:
      throw PythonBackendException("Unknown metric kind");
  }
}

void
PbMetricFamily::AddMetric(std::unique_ptr<CustomMetric> metric)
{
  std::lock_guard<std::mutex> lock(metric_map_mu_);
  if (metric_map_.find(metric->Labels()) != metric_map_.end()) {
    throw PythonBackendException("Metric with the same labels already exists.");
  } else {
    std::unique_ptr<PbMetric> pb_metric =
        std::make_unique<PbMetric>(metric->Labels(), metric_family_);
    metric_map_.insert({metric->Labels(), std::move(pb_metric)});
  }
}

std::unordered_map<std::string, std::unique_ptr<PbMetric>>*
PbMetricFamily::MetricMap()
{
  return &metric_map_;
}

MetricKind
PbMetricFamily::Kind()
{
  return kind_;
}

}}}  // namespace triton::backend::python
