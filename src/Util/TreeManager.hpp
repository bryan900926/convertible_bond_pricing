#pragma once

#include "Pricing/CbModel.h"
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <stdexcept>

class TreeManager {
    std::string _data_path;
    float _max_size_gb;
    std::vector<std::vector<PackedNode>> _saved_tree;
    std::vector<int> _node_cnts; 
    size_t _current_buffer_nodes = 0; // FIX: Tracks actual node count in RAM
    std::ofstream _file_stream;

    // --- Reading Variables ---
    std::ifstream _in_stream;
    std::vector<std::streampos> _period_offsets; // Stores the exact byte location of each period
    std::vector<std::vector<PackedNode>> _read_buffer;
    int _buffer_start_idx = -1; // The lowest period 'i' currently in RAM
    int _buffer_end_idx = -1;   // The highest period 'i' currently in RAM

    void save_tree(const std::vector<PackedNode>& tree_data) {
        if (_file_stream) {
            size_t num_elements = tree_data.size();
            _file_stream.write(reinterpret_cast<const char*>(&num_elements), sizeof(size_t));
            if (num_elements > 0) {
                _file_stream.write(reinterpret_cast<const char*>(tree_data.data()), num_elements * sizeof(PackedNode));
            }
        } else {
            throw std::runtime_error("Unable to open file for writing: " + _data_path);
        }
    }

public:
    TreeManager(const std::string &path, const float gigabyte)
        : _data_path(path), _max_size_gb(gigabyte) {
        if (std::filesystem::exists(_data_path)) {
            std::filesystem::remove(_data_path);
        } else {
            std::filesystem::create_directories(std::filesystem::path(_data_path).parent_path());
        }
        _file_stream.open(_data_path, std::ios::binary | std::ios::out | std::ios::app);
    }

    // --- FORWARD PASS (Writing) ---

    void append_tree(const std::vector<PackedNode>& tree_data) {
        _saved_tree.push_back(tree_data);
        _node_cnts.push_back(tree_data.size());
        _current_buffer_nodes += tree_data.size(); // Keep running total of nodes

        // FIX: Calculate memory based on node count, not period count
        double current_size_gb = (_current_buffer_nodes * sizeof(PackedNode)) / (1024.0 * 1024.0 * 1024.0);

        if (current_size_gb >= _max_size_gb) {
            for (const auto &data : _saved_tree) {
                save_tree(data);
            }
            _saved_tree.clear();
            _current_buffer_nodes = 0;
        }
    }

    void complete_save() {
        if (!_saved_tree.empty()) {
            for (const auto &data : _saved_tree) {
                save_tree(data);
            }
            _saved_tree.clear();
            _current_buffer_nodes = 0;
        }
    }

    // --- BACKWARD PASS (Reading) ---

    // Call this once before your backward `for` loop begins
    void prepare_for_reading() {
        complete_save();      // Ensure everything is flushed
        _file_stream.close(); // Close the output stream
        
        _in_stream.open(_data_path, std::ios::binary);
        if (!_in_stream) {
            throw std::runtime_error("Unable to open file for reading: " + _data_path);
        }

        // Calculate file offsets for every period based on _node_cnts
        _period_offsets.resize(_node_cnts.size());
        std::streampos current_pos = 0;
        for (size_t i = 0; i < _node_cnts.size(); ++i) {
            _period_offsets[i] = current_pos;
            current_pos += sizeof(size_t) + _node_cnts[i] * sizeof(PackedNode);
        }
    }

    // Call this inside your backward loop for period 'i'
    const std::vector<PackedNode>& get_period(int target_i) {
        // If the period is not currently loaded in our RAM batch, load a new batch
        if (target_i < _buffer_start_idx || target_i > _buffer_end_idx) {
            load_batch_backwards(target_i);
        }
        
        // Return the specific period from the RAM batch
        return _read_buffer[target_i - _buffer_start_idx];
    }

    void close() {
        if (_in_stream.is_open()) {
            _in_stream.close();
        }
    }

private:
    void load_batch_backwards(int target_i) {
        const double max_bytes = _max_size_gb * 1024.0 * 1024.0 * 1024.0;
        double current_batch_bytes = 0;

        // Figure out how far back (start_i) we can read without exceeding RAM limit
        int start_i = target_i;
        while (start_i > 0) {
            double period_bytes = sizeof(size_t) + _node_cnts[start_i - 1] * sizeof(PackedNode);
            
            // Stop expanding batch if adding this period breaches max size 
            // (Unless it's the very first period we are trying to add)
            if (current_batch_bytes + period_bytes > max_bytes && start_i != target_i) {
                break; 
            }
            current_batch_bytes += period_bytes;
            start_i--;
        }
        start_i++; // Adjust to inclusive start index

        _buffer_start_idx = start_i;
        _buffer_end_idx = target_i;
        
        // Allocate space for the batch
        _read_buffer.resize(target_i - start_i + 1);

        // Seek directly to the beginning of this batch on the hard drive
        _in_stream.seekg(_period_offsets[start_i - 1]);

        // Read the batch sequentially into RAM
        for (int i = start_i; i <= target_i; ++i) {
            size_t num_elements = 0;
            _in_stream.read(reinterpret_cast<char*>(&num_elements), sizeof(size_t));
            
            _read_buffer[i - start_i].resize(num_elements);
            if (num_elements > 0) {
                _in_stream.read(reinterpret_cast<char*>(_read_buffer[i - start_i].data()), 
                                num_elements * sizeof(PackedNode));
            }
        }
        
    }
};