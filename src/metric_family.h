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
#include "metric.h"
#include "pb_string.h"
#include "pb_utils.h"

#ifdef TRITON_PB_STUB
#include <pybind11/embed.h>
namespace py = pybind11;
#else
#include "triton/core/tritonserver.h"
#endif

namespace triton { namespace backend { namespace python {

enum MetricFamilyRequestKind { MetricFamilyNew, MetricFamilyDelete };

struct MetricFamilyShm {
  bi::managed_external_buffer::handle_t name_shm_handle;
  bi::managed_external_buffer::handle_t description_shm_handle;
  bi::managed_external_buffer::handle_t key_name_shm_handle;
  MetricKind kind;
  MetricFamilyRequestKind metric_family_request_kind;
  void* metric_family_address;
};

class MetricFamily {
 public:
  MetricFamily(
      const std::string& name, const std::string& description,
      const MetricKind& kind);

  ~MetricFamily();

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
  static std::unique_ptr<MetricFamily> LoadFromSharedMemory(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      bi::managed_external_buffer::handle_t handle);

  /// Get the address of the TRITONSERVER_MetricFamily object.
  /// \return Returns the address of the TRITONSERVER_MetricFamily object.
  void* MetricFamilyAddress();

#ifdef TRITON_PB_STUB
  /// Store the metric in the metric map.
  /// \param metric The Metric to be added.
  void AddMetric(std::shared_ptr<Metric> metric);
#else
  /// Initialize the TRITONSERVER_MetricFamily object.
  /// \return Returns the address of the TRITONSERVER_MetricFamily object.
  void* InitializeTritonMetricFamily();

  /// Helper function to convert the MetricKind enum to TRITONSERVER_MetricKind
  /// \param kind The MetricKind enum to be converted.
  /// \return Returns the TRITONSERVER_MetricKind enum.
  TRITONSERVER_MetricKind ToTritonServerMetricKind(const MetricKind& kind);

  /// Clear the TRITONSERVER_MetricFamily object.
  void ClearTritonMetricFamily();
#endif

  /// Disallow copying the metric family object.
  DISALLOW_COPY_AND_ASSIGN(MetricFamily);

 private:
  MetricFamily(
      AllocatedSharedMemory<char>& custom_metric_family_shm,
      std::unique_ptr<PbString>& name_shm,
      std::unique_ptr<PbString>& description_shm);
  std::string name_;
  std::string description_;
  std::string unique_name_;
  MetricKind kind_;
  MetricFamilyRequestKind metric_family_request_kind_;
  void* metric_family_address_;
  std::mutex metric_map_mu_;
  // Need to keep track of the metrics associated with the metric family to make
  // sure the metrics are cleaned up before the metric family is deleted.
  std::unordered_map<void*, std::shared_ptr<Metric>> metric_map_;

  // Shared Memory Data Structures
  AllocatedSharedMemory<char> custom_metric_family_shm_;
  MetricFamilyShm* custom_metric_family_shm_ptr_;
  bi::managed_external_buffer::handle_t shm_handle_;
  std::unique_ptr<PbString> name_shm_;
  std::unique_ptr<PbString> description_shm_;
};

}}};  // namespace triton::backend::python
