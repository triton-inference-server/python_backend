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

#include "metric.h"

#ifdef TRITON_PB_STUB
#include "pb_stub.h"
#endif

namespace triton { namespace backend { namespace python {

Metric::Metric(
    const std::string& family_name, const std::string& labels,
    void* metric_address)
    : family_name_(family_name), labels_(labels), value_(0),
      metric_request_kind_(MetricNew), metric_address_(nullptr),
      metric_family_address_(metric_address)
{
#ifdef TRITON_PB_STUB
  // Send the request to create the Metric to the parent process
  std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
  SaveToSharedMemory(stub->ShmPool());
  CustomMetricsMessage* custom_metrics_msg = nullptr;
  try {
    stub->SendCustomMetricsMessage(
        &custom_metrics_msg, PYTHONSTUB_MetricRequest, shm_handle_);
  }
  catch (const PythonBackendException& pb_exception) {
    throw PythonBackendException(
        "Error when creating Metric: " + std::string(pb_exception.what()));
  }
  metric_address_ = custom_metrics_msg->address;
#endif
}

Metric::~Metric()
{
#ifdef TRITON_PB_STUB
  Clear();
#endif
}

const std::string&
Metric::FamilyName()
{
  return family_name_;
}

const std::string&
Metric::Labels()
{
  return labels_;
}

bi::managed_external_buffer::handle_t
Metric::ShmHandle()
{
  return shm_handle_;
}

const MetricRequestKind&
Metric::RequestKind()
{
  return metric_request_kind_;
}

double
Metric::Value()
{
  return value_;
}

void
Metric::SaveToSharedMemory(std::unique_ptr<SharedMemoryManager>& shm_pool)
{
  AllocatedSharedMemory<char> custom_metric_shm =
      shm_pool->Construct<char>(sizeof(MetricShm), true);
  custom_metric_shm_ptr_ =
      reinterpret_cast<MetricShm*>(custom_metric_shm.data_.get());

  std::unique_ptr<PbString> family_name_shm =
      PbString::Create(shm_pool, FamilyName());
  std::unique_ptr<PbString> labels_shm = PbString::Create(shm_pool, Labels());

  custom_metric_shm_ptr_->value = value_;
  custom_metric_shm_ptr_->metric_request_kind = metric_request_kind_;
  custom_metric_shm_ptr_->family_name_shm_handle = family_name_shm->ShmHandle();
  custom_metric_shm_ptr_->labels_shm_handle = labels_shm->ShmHandle();
  custom_metric_shm_ptr_->metric_family_address = metric_family_address_;
  custom_metric_shm_ptr_->metric_address = metric_address_;

  // Save the references to shared memory.
  custom_metric_shm_ = std::move(custom_metric_shm);
  family_name_shm_ = std::move(family_name_shm);
  labels_shm_ = std::move(labels_shm);
  shm_handle_ = custom_metric_shm.handle_;
}

std::unique_ptr<Metric>
Metric::LoadFromSharedMemory(
    std::unique_ptr<SharedMemoryManager>& shm_pool,
    bi::managed_external_buffer::handle_t handle)
{
  AllocatedSharedMemory<char> custom_metric_shm = shm_pool->Load<char>(handle);
  MetricShm* custom_metric_shm_ptr =
      reinterpret_cast<MetricShm*>(custom_metric_shm.data_.get());

  std::unique_ptr<PbString> family_name_shm = PbString::LoadFromSharedMemory(
      shm_pool, custom_metric_shm_ptr->family_name_shm_handle);
  std::unique_ptr<PbString> labels_shm = PbString::LoadFromSharedMemory(
      shm_pool, custom_metric_shm_ptr->labels_shm_handle);

  return std::unique_ptr<Metric>(
      new Metric(custom_metric_shm, family_name_shm, labels_shm));
}

Metric::Metric(
    AllocatedSharedMemory<char>& custom_metric_shm,
    std::unique_ptr<PbString>& family_name_shm,
    std::unique_ptr<PbString>& labels_shm)
    : custom_metric_shm_(std::move(custom_metric_shm)),
      family_name_shm_(std::move(family_name_shm)),
      labels_shm_(std::move(labels_shm))
{
  custom_metric_shm_ptr_ =
      reinterpret_cast<MetricShm*>(custom_metric_shm_.data_.get());
  family_name_ = family_name_shm_->String();
  labels_ = labels_shm_->String();
  value_ = custom_metric_shm_ptr_->value;
  metric_request_kind_ = custom_metric_shm_ptr_->metric_request_kind;
  metric_family_address_ = custom_metric_shm_ptr_->metric_family_address;
  metric_address_ = custom_metric_shm_ptr_->metric_address;
}

void*
Metric::MetricAddress()
{
  return metric_address_;
}

#ifdef TRITON_PB_STUB
void
Metric::SendIncrementRequest(const double& value)
{
  std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
  metric_request_kind_ = MetricIncrement;
  value_ = value;
  SaveToSharedMemory(stub->ShmPool());
  CustomMetricsMessage* custom_metrics_msg = nullptr;
  try {
    stub->SendCustomMetricsMessage(
        &custom_metrics_msg, PYTHONSTUB_MetricRequest, shm_handle_);
  }
  catch (const PythonBackendException& pb_exception) {
    throw PythonBackendException(
        "Failed to increment metric value: " +
        std::string(pb_exception.what()));
  }
}

void
Metric::SendSetValueRequest(const double& value)
{
  std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
  metric_request_kind_ = MetricSet;
  value_ = value;
  SaveToSharedMemory(stub->ShmPool());
  CustomMetricsMessage* custom_metrics_msg = nullptr;
  try {
    stub->SendCustomMetricsMessage(
        &custom_metrics_msg, PYTHONSTUB_MetricRequest, shm_handle_);
  }
  catch (const PythonBackendException& pb_exception) {
    throw PythonBackendException(
        "Failed to set metric value: " + std::string(pb_exception.what()));
  }
}

double
Metric::SendGetValueRequest()
{
  std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
  metric_request_kind_ = MetricValue;
  SaveToSharedMemory(stub->ShmPool());
  CustomMetricsMessage* custom_metrics_msg = nullptr;
  try {
    stub->SendCustomMetricsMessage(
        &custom_metrics_msg, PYTHONSTUB_MetricRequest, shm_handle_);
  }
  catch (const PythonBackendException& pb_exception) {
    throw PythonBackendException(
        "Failed to get metric value: " + std::string(pb_exception.what()));
  }

  return custom_metrics_msg->value;
}

void
Metric::Clear()
{
  if (metric_request_kind_ != MetricDelete) {
    metric_request_kind_ = MetricDelete;
    std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
    SaveToSharedMemory(stub->ShmPool());
    CustomMetricsMessage* custom_metrics_msg = nullptr;
    try {
      stub->SendCustomMetricsMessage(
          &custom_metrics_msg, PYTHONSTUB_MetricRequest, shm_handle_);
    }
    catch (const PythonBackendException& pb_exception) {
      std::cerr << "Error when deleting Metric: " << pb_exception.what()
                << "\n";
    }
  }
}

#else
void*
Metric::InitializeTritonMetric()
{
  std::vector<const TRITONSERVER_Parameter*> labels_params;
  ParseLabels(labels_params, labels_);
  TRITONSERVER_Metric* triton_metric = nullptr;
  THROW_IF_TRITON_ERROR(TRITONSERVER_MetricNew(
      &triton_metric,
      reinterpret_cast<TRITONSERVER_MetricFamily*>(metric_family_address_),
      labels_params.data(), labels_params.size()));
  for (const auto label : labels_params) {
    TRITONSERVER_ParameterDelete(const_cast<TRITONSERVER_Parameter*>(label));
  }
  return reinterpret_cast<void*>(triton_metric);
}

void
Metric::ParseLabels(
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
Metric::HandleMetricOperation(CustomMetricsMessage** metrics_message_ptr)
{
  if (metric_request_kind_ == MetricValue) {
    UpdateValue();
    (*metrics_message_ptr)->value = value_;
  } else if (metric_request_kind_ == MetricIncrement) {
    Increment(value_);
  } else if (metric_request_kind_ == MetricSet) {
    SetValue(value_);
  } else {
    throw PythonBackendException("Unknown metric operation");
  }
}

void
Metric::Increment(const double& value)
{
  auto triton_metric = reinterpret_cast<TRITONSERVER_Metric*>(metric_address_);
  THROW_IF_TRITON_ERROR(TRITONSERVER_MetricIncrement(triton_metric, value));
  UpdateValue();
}

void
Metric::SetValue(const double& value)
{
  auto triton_metric = reinterpret_cast<TRITONSERVER_Metric*>(metric_address_);
  THROW_IF_TRITON_ERROR(TRITONSERVER_MetricSet(triton_metric, value));
  UpdateValue();
}

void
Metric::UpdateValue()
{
  auto triton_metric = reinterpret_cast<TRITONSERVER_Metric*>(metric_address_);
  THROW_IF_TRITON_ERROR(TRITONSERVER_MetricValue(triton_metric, &value_));
}

void
Metric::ClearTritonMetric()
{
  auto metric = reinterpret_cast<TRITONSERVER_Metric*>(metric_address_);
  if (metric != nullptr) {
    LOG_IF_ERROR(TRITONSERVER_MetricDelete(metric), "deleting metric");
  }
}

#endif

}}}  // namespace triton::backend::python
