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
#include <unordered_map>
#include "ipc_message.h"
#include "pb_custom_metrics.h"
#include "pb_string.h"
#include "pb_utils.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace backend { namespace python {

class MetricRegistry {
 public:
  MetricRegistry(
      const std::string& label, TRITONSERVER_MetricFamily* metric_family);

  ~MetricRegistry();

  // Parse the labels string into a vector of TRITONSERVER_Parameter
  void ParseLabels(
      std::vector<const TRITONSERVER_Parameter*>& labels_params,
      const std::string& labels);

  // Get the labels of the metric
  const std::string& Labels();

  // Increment the value of the metric by the given value
  void Increment(const double& value);

  // Set the value of the metric
  void SetValue(const double& value);

  // Get the value of the metric
  double Value();

  // Update the value of the metric
  void UpdateValue();

 private:
  // The labels of the metric serve as the key to the metric map
  std::string labels_;
  double value_;
  TRITONSERVER_Metric* metric_;
};

class MetricFamilyRegistry {
 public:
  MetricFamilyRegistry(
      const std::string& name, const std::string& description,
      const MetricKind& kind);

  ~MetricFamilyRegistry();

  // Helper function to convert the MetricKind enum to TRITONSERVER_MetricKind
  TRITONSERVER_MetricKind ToTritonServerMetricKind(const MetricKind& kind);

  // Get the TRITONSERVER_MetricFamily object
  TRITONSERVER_MetricFamily* MetricFamily() { return metric_family_; }

  // Add a metric to the metric family and store it in the metric map
  void AddMetric(std::unique_ptr<PbCustomMetric> metric);

  // Get the metric map
  std::unordered_map<std::string, std::unique_ptr<MetricRegistry>>* MetricMap();

  // Get the metric kind
  MetricKind Kind();

 private:
  std::string name_;
  std::string description_;
  MetricKind kind_;
  TRITONSERVER_MetricFamily* metric_family_;
  std::mutex metric_map_mu_;
  std::unordered_map<std::string, std::unique_ptr<MetricRegistry>> metric_map_;
};

}}};  // namespace triton::backend::python
