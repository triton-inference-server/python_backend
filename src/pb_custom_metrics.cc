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

#include "pb_custom_metrics.h"

#ifdef TRITON_PB_STUB
#include "pb_stub.h"
#endif

namespace triton { namespace backend { namespace python {

CustomMetric::CustomMetric(
    const std::string& family_name, const std::string& labels)
    : family_name_(family_name), labels_(labels), value_(0),
      metric_request_kind_(MetricNew)
{
}

CustomMetric::~CustomMetric()
{
#ifdef TRITON_PB_STUB
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
      std::cerr << "Error when deleting CustomMetric: " << pb_exception.what()
                << "\n";
    }
    stub->ClearMetric(family_name_, labels_);
  }
#endif
}

const std::string&
CustomMetric::FamilyName()
{
  return family_name_;
}

const std::string&
CustomMetric::Labels()
{
  return labels_;
}

bi::managed_external_buffer::handle_t
CustomMetric::ShmHandle()
{
  return shm_handle_;
}

const MetricRequestKind&
CustomMetric::RequestKind()
{
  return metric_request_kind_;
}

double
CustomMetric::Value()
{
  return value_;
}

void
CustomMetric::SaveToSharedMemory(std::unique_ptr<SharedMemoryManager>& shm_pool)
{
  AllocatedSharedMemory<char> custom_metric_shm =
      shm_pool->Construct<char>(sizeof(CustomMetricShm), true);
  custom_metric_shm_ptr_ =
      reinterpret_cast<CustomMetricShm*>(custom_metric_shm.data_.get());

  std::unique_ptr<PbString> family_name_shm =
      PbString::Create(shm_pool, FamilyName());
  std::unique_ptr<PbString> labels_shm = PbString::Create(shm_pool, Labels());

  custom_metric_shm_ptr_->value = value_;
  custom_metric_shm_ptr_->metric_request_kind = metric_request_kind_;
  custom_metric_shm_ptr_->family_name_shm_handle = family_name_shm->ShmHandle();
  custom_metric_shm_ptr_->labels_shm_handle = labels_shm->ShmHandle();

  // Save the references to shared memory.
  custom_metric_shm_ = std::move(custom_metric_shm);
  family_name_shm_ = std::move(family_name_shm);
  labels_shm_ = std::move(labels_shm);
  shm_handle_ = custom_metric_shm.handle_;
}

std::unique_ptr<CustomMetric>
CustomMetric::LoadFromSharedMemory(
    std::unique_ptr<SharedMemoryManager>& shm_pool,
    bi::managed_external_buffer::handle_t handle)
{
  AllocatedSharedMemory<char> custom_metric_shm = shm_pool->Load<char>(handle);
  CustomMetricShm* custom_metric_shm_ptr =
      reinterpret_cast<CustomMetricShm*>(custom_metric_shm.data_.get());

  std::unique_ptr<PbString> family_name_shm = PbString::LoadFromSharedMemory(
      shm_pool, custom_metric_shm_ptr->family_name_shm_handle);
  std::unique_ptr<PbString> labels_shm = PbString::LoadFromSharedMemory(
      shm_pool, custom_metric_shm_ptr->labels_shm_handle);

  return std::unique_ptr<CustomMetric>(
      new CustomMetric(custom_metric_shm, family_name_shm, labels_shm));
}

CustomMetric::CustomMetric(
    AllocatedSharedMemory<char>& custom_metric_shm,
    std::unique_ptr<PbString>& family_name_shm,
    std::unique_ptr<PbString>& labels_shm)
    : custom_metric_shm_(std::move(custom_metric_shm)),
      family_name_shm_(std::move(family_name_shm)),
      labels_shm_(std::move(labels_shm))
{
  custom_metric_shm_ptr_ =
      reinterpret_cast<CustomMetricShm*>(custom_metric_shm_.data_.get());
  family_name_ = family_name_shm_->String();
  labels_ = labels_shm_->String();
  value_ = custom_metric_shm_ptr_->value;
  metric_request_kind_ = custom_metric_shm_ptr_->metric_request_kind;
}

CustomMetricFamily::CustomMetricFamily(
    const std::string& name, const std::string& description,
    const MetricKind& kind)
    : name_(name), description_(description), kind_(kind),
      metric_family_request_kind_(MetricFamilyNew)
{
}

CustomMetricFamily::~CustomMetricFamily()
{
#ifdef TRITON_PB_STUB
  if (metric_family_request_kind_ != MetricFamilyDelete) {
    metric_family_request_kind_ = MetricFamilyDelete;
    std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
    SaveToSharedMemory(stub->ShmPool());
    CustomMetricsMessage* custom_metrics_msg = nullptr;
    try {
      stub->SendCustomMetricsMessage(
          &custom_metrics_msg, PYTHONSTUB_MetricRequest, shm_handle_);
    }
    catch (const PythonBackendException& pb_exception) {
      std::cerr << "Error when deleting CustomMetricFamily: "
                << pb_exception.what() << "\n";
    }
    stub->ClearMetricFamily(name_);
  }
#endif
};

const std::string&
CustomMetricFamily::Name()
{
  return name_;
}

const std::string&
CustomMetricFamily::Description()
{
  return description_;
}

const MetricKind&
CustomMetricFamily::Kind()
{
  return kind_;
}

bi::managed_external_buffer::handle_t
CustomMetricFamily::ShmHandle()
{
  return shm_handle_;
}

const MetricFamilyRequestKind&
CustomMetricFamily::RequestKind()
{
  return metric_family_request_kind_;
}

void
CustomMetricFamily::SaveToSharedMemory(
    std::unique_ptr<SharedMemoryManager>& shm_pool)
{
  AllocatedSharedMemory<char> custom_metric_family_shm =
      shm_pool->Construct<char>(sizeof(CustomMetricFamilyShm));

  custom_metric_family_shm_ptr_ = reinterpret_cast<CustomMetricFamilyShm*>(
      custom_metric_family_shm.data_.get());
  std::unique_ptr<PbString> name_shm = PbString::Create(shm_pool, Name());
  std::unique_ptr<PbString> description_shm =
      PbString::Create(shm_pool, Description());

  custom_metric_family_shm_ptr_->kind = kind_;
  custom_metric_family_shm_ptr_->metric_family_request_kind =
      metric_family_request_kind_;
  custom_metric_family_shm_ptr_->name_shm_handle = name_shm->ShmHandle();
  custom_metric_family_shm_ptr_->description_shm_handle =
      description_shm->ShmHandle();

  // Save the references to shared memory.
  custom_metric_family_shm_ = std::move(custom_metric_family_shm);
  name_shm_ = std::move(name_shm);
  description_shm_ = std::move(description_shm);
  shm_handle_ = custom_metric_family_shm.handle_;
}

std::unique_ptr<CustomMetricFamily>
CustomMetricFamily::LoadFromSharedMemory(
    std::unique_ptr<SharedMemoryManager>& shm_pool,
    bi::managed_external_buffer::handle_t handle)
{
  AllocatedSharedMemory<char> custom_metric_family_shm =
      shm_pool->Load<char>(handle);
  CustomMetricFamilyShm* custom_metric_family_shm_ptr =
      reinterpret_cast<CustomMetricFamilyShm*>(
          custom_metric_family_shm.data_.get());
  std::unique_ptr<PbString> name_shm = PbString::LoadFromSharedMemory(
      shm_pool, custom_metric_family_shm_ptr->name_shm_handle);
  std::unique_ptr<PbString> description_shm = PbString::LoadFromSharedMemory(
      shm_pool, custom_metric_family_shm_ptr->description_shm_handle);

  return std::unique_ptr<CustomMetricFamily>(new CustomMetricFamily(
      custom_metric_family_shm, name_shm, description_shm));
}

CustomMetricFamily::CustomMetricFamily(
    AllocatedSharedMemory<char>& custom_metric_family_shm,
    std::unique_ptr<PbString>& name_shm,
    std::unique_ptr<PbString>& description_shm)
    : custom_metric_family_shm_(std::move(custom_metric_family_shm)),
      name_shm_(std::move(name_shm)),
      description_shm_(std::move(description_shm))
{
  custom_metric_family_shm_ptr_ = reinterpret_cast<CustomMetricFamilyShm*>(
      custom_metric_family_shm_.data_.get());
  name_ = name_shm_->String();
  description_ = description_shm_->String();
  kind_ = custom_metric_family_shm_ptr_->kind;
  metric_family_request_kind_ =
      custom_metric_family_shm_ptr_->metric_family_request_kind;
}

void
CustomMetricFamily::ClearMetric(const std::string& labels)
{
  std::lock_guard<std::mutex> lock(metric_map_mu_);
  metric_map_.erase(labels);
}

#ifdef TRITON_PB_STUB
void
CustomMetric::Increment(const double& value)
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
        "Failed to increment value: " + std::string(pb_exception.what()));
  }
}

void
CustomMetric::SetValue(const double& value)
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
        "Failed to set value: " + std::string(pb_exception.what()));
  }
}

double
CustomMetric::GetValue()
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
        "Failed to get value: " + std::string(pb_exception.what()));
  }

  return custom_metrics_msg->value;
}

std::shared_ptr<CustomMetric>
CustomMetricFamily::CreateMetric(const std::string& labels_str)
{
  std::lock_guard<std::mutex> lock(metric_map_mu_);
  if (metric_map_.find(labels_str) != metric_map_.end()) {
    // If the metric already exists, return the existing one.
    std::cerr << "Metric '" << labels_str
              << "' already exists in Metric family '" << name_ << "' .\n";
  } else {
    auto metric =
        std::make_shared<CustomMetric>(name_ + description_, labels_str);
    metric_map_[labels_str] = metric;
    std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
    metric->SaveToSharedMemory(stub->ShmPool());
    CustomMetricsMessage* custom_metrics_msg = nullptr;
    try {
      stub->SendCustomMetricsMessage(
          &custom_metrics_msg, PYTHONSTUB_MetricRequest, metric->ShmHandle());
    }
    catch (const PythonBackendException& pb_exception) {
      throw PythonBackendException(
          "Failed to send metric message: " + std::string(pb_exception.what()));
    }
  }
  return metric_map_[labels_str];
}
#endif

}}}  // namespace triton::backend::python
