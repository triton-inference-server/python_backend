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

#pragma once

#include <string>
#include "ipc_message.h"
#include "pb_string.h"
#include "pb_utils.h"

#ifdef TRITON_PB_STUB
#include <pybind11/embed.h>
namespace py = pybind11;
#endif

namespace triton { namespace backend { namespace python {

struct PbCustomMetricShm {
  bi::managed_external_buffer::handle_t family_name_shm_handle;
  bi::managed_external_buffer::handle_t labels_shm_handle;
  double value;
  MetricRequestKind metric_request_kind;
};

class PbCustomMetric {
 public:
  PbCustomMetric(const std::string& family_name, const std::string& labels);

  ~PbCustomMetric();

  const std::string& FamilyName();
  const std::string& Labels();
  bi::managed_external_buffer::handle_t ShmHandle();
  const MetricRequestKind& RequestKind();
  double Value();

  /// Save Custom Metric object to shared memory.
  /// \param shm_pool Shared memory pool to save the custom metric object.
  void SaveToSharedMemory(std::unique_ptr<SharedMemoryManager>& shm_pool);

  /// Create a Custom Metric object from shared memory.
  /// \param shm_pool Shared memory pool
  /// \param handle Shared memory handle of the custom metric.
  /// \return Returns the custom metrics in the specified request_handle
  /// location.
  static std::unique_ptr<PbCustomMetric> LoadFromSharedMemory(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      bi::managed_external_buffer::handle_t handle);

#ifdef TRITON_PB_STUB
  /// Increment the value of the metric by the specified value.
  /// \param value The value to increment the metric by.
  void Increment(const double& value);

  /// Set the value of the metric to the specified value.
  /// \param value The value to set the metric to.
  void SetValue(const double& value);

  /// Get the value of the metric.
  /// \return Returns the value of the metric.
  double GetValue();
#endif

  /// Disallow copying the custom metric object.
  DISALLOW_COPY_AND_ASSIGN(PbCustomMetric);

 private:
  PbCustomMetric(
      AllocatedSharedMemory<char>& custom_metric_shm,
      std::unique_ptr<PbString>& family_name_shm,
      std::unique_ptr<PbString>& labels_shm);
  std::string family_name_;
  std::string labels_;
  double value_;
  MetricRequestKind metric_request_kind_;

  // Shared Memory Data Structures
  AllocatedSharedMemory<char> custom_metric_shm_;
  PbCustomMetricShm* custom_metric_shm_ptr_;
  bi::managed_external_buffer::handle_t shm_handle_;
  std::unique_ptr<PbString> family_name_shm_;
  std::unique_ptr<PbString> labels_shm_;
};

struct PbCustomMetricFamilyShm {
  bi::managed_external_buffer::handle_t name_shm_handle;
  bi::managed_external_buffer::handle_t description_shm_handle;
  bi::managed_external_buffer::handle_t key_name_shm_handle;
  MetricKind kind;
  MetricFamilyRequestKind metric_family_request_kind;
};

class PbCustomMetricFamily {
 public:
  PbCustomMetricFamily(
      const std::string& name, const std::string& description,
      const MetricKind& kind);

  ~PbCustomMetricFamily();

  /// Get the name of the custom metric family.
  /// \return Returns the name of the metric family.
  const std::string& Name();

  /// Get the description of the custom metric family.
  /// \return Returns the description of the custom metric family.
  const std::string& Description();

  /// Get the metric kind of the custom metric family.
  /// \return Returns the metric kind of the custom metric family.
  const MetricKind& Kind();

  /// Get the shm handle of the custom metric family.
  /// \return Returns the shm handle of the custom metric family.
  bi::managed_external_buffer::handle_t ShmHandle();

  /// Get the custom metric family request kind.
  /// \return Returns the custom metric family request kind.
  const MetricFamilyRequestKind& RequestKind();

  /// Save a custom metric family to shared memory.
  /// \param shm_pool Shared memory pool to save the custom metric family.
  void SaveToSharedMemory(std::unique_ptr<SharedMemoryManager>& shm_pool);

  /// Create a Custom Metric Family object from shared memory.
  /// \param shm_pool Shared memory pool
  /// \param handle Shared memory handle of the custom metric family.
  /// \return Returns the custom metric family in the specified request_handle
  /// location.
  static std::unique_ptr<PbCustomMetricFamily> LoadFromSharedMemory(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      bi::managed_external_buffer::handle_t handle);

  void ClearMetric(const std::string& labels);

#ifdef TRITON_PB_STUB
  /// Create a custom metric object with the specified labels and store it in
  /// the metric_map_.
  /// \param labels The labels for the custom metric.
  /// \return Returns a custom metric object for the specified labels.
  std::shared_ptr<PbCustomMetric> CreateMetric(const std::string& labels);
#endif

  /// Disallow copying the custom metric family object.
  DISALLOW_COPY_AND_ASSIGN(PbCustomMetricFamily);

 private:
  PbCustomMetricFamily(
      AllocatedSharedMemory<char>& custom_metric_family_shm,
      std::unique_ptr<PbString>& name_shm,
      std::unique_ptr<PbString>& description_shm);
  std::string name_;
  std::string description_;
  std::string unique_name_;
  MetricKind kind_;
  MetricFamilyRequestKind metric_family_request_kind_;

  // Shared Memory Data Structures
  AllocatedSharedMemory<char> custom_metric_family_shm_;
  PbCustomMetricFamilyShm* custom_metric_family_shm_ptr_;
  bi::managed_external_buffer::handle_t shm_handle_;
  std::unique_ptr<PbString> name_shm_;
  std::unique_ptr<PbString> description_shm_;

  std::mutex metric_map_mu_;
  std::unordered_map<std::string, std::shared_ptr<PbCustomMetric>> metric_map_;
};

}}};  // namespace triton::backend::python
