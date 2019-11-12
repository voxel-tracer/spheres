#pragma once

#define RATIO(x,a)  (100.0 * x / a)

struct multi_iter_warp_counter {
    int print_out_iter;
    int max_in_iter;

    int* out_iter;
    int* in_iter;

    int* in_max;

    unsigned long long* total;
    unsigned long long* by25;
    unsigned long long* by50;
    unsigned long long* by75;
    unsigned long long* by100;

    __host__ multi_iter_warp_counter() {}
    __host__ multi_iter_warp_counter(int max, int print) : max_in_iter(max), print_out_iter(print) {}

    __host__ void allocateDeviceMem() {
        checkCudaErrors(cudaMalloc((void**)& total, max_in_iter * sizeof(unsigned long long)));
        checkCudaErrors(cudaMalloc((void**)& by25, max_in_iter * sizeof(unsigned long long)));
        checkCudaErrors(cudaMalloc((void**)& by50, max_in_iter * sizeof(unsigned long long)));
        checkCudaErrors(cudaMalloc((void**)& by75, max_in_iter * sizeof(unsigned long long)));
        checkCudaErrors(cudaMalloc((void**)& by100, max_in_iter * sizeof(unsigned long long)));

        checkCudaErrors(cudaMalloc((void**)& in_iter, sizeof(int)));
        checkCudaErrors(cudaMalloc((void**)& out_iter, sizeof(int)));
        checkCudaErrors(cudaMalloc((void**)& in_max, sizeof(int)));
    }

    __host__ void freeDeviceMem() const {
        checkCudaErrors(cudaFree(total));
        checkCudaErrors(cudaFree(by25));
        checkCudaErrors(cudaFree(by50));
        checkCudaErrors(cudaFree(by75));
        checkCudaErrors(cudaFree(by100));
        checkCudaErrors(cudaFree(in_iter));
        checkCudaErrors(cudaFree(out_iter));
        checkCudaErrors(cudaFree(in_max));
    }

    __device__ void reset(int pid, bool first) {
        if (first) {
            if (pid < max_in_iter) {
                total[pid] = 0;
                by25[pid] = 0;
                by50[pid] = 0;
                by75[pid] = 0;
                by100[pid] = 0;
            }
            in_max[0] = 0;
            out_iter[0] = 0;
            in_iter[0] = 0;
        }
        if (pid == 0)
            out_iter[0]++;
    }

    __device__ void increment(int in_it, int lane_id) {
        if (out_iter[0] != print_out_iter)
            return;

        atomicMax(in_max, in_it);

        if (in_it >= max_in_iter)
            return;

        atomicMax(in_iter, in_it);

        // first active thread of the warp should increment the metrics
        const int num_active = __popc(__activemask());
        const int idx_lane = __popc(__activemask() & ((1u << lane_id) - 1));
        if (idx_lane == 0) {
            atomicAdd(total + in_it, 1);
            if (num_active == 32)
                atomicAdd(by100 + in_it, 1);
            else if (num_active >= 24)
                atomicAdd(by75 + in_it, 1);
            else if (num_active >= 16)
                atomicAdd(by50 + in_it, 1);
            else if (num_active >= 8)
                atomicAdd(by25 + in_it, 1);
        }
    }

    __device__ void print() const {
        if (out_iter[0] != print_out_iter)
            return;

        for (int i = 0; i <= in_iter[0]; i++) {
            unsigned long long tot = total[i];
            if (tot > 0) {
                unsigned long long num100 = by100[i];
                unsigned long long num75 = by75[i];
                unsigned long long num50 = by50[i];
                unsigned long long num25 = by25[i];
                unsigned long long less25 = tot - num100 - num75 - num50 - num25;
                printf("iteration %4d: total %7llu, 100%% %6.2f%%, >=75%% %6.2f%%, >=50%% %6.2f%%, >=25%% %6.2f%%, less %6.2f%%\n", i, tot,
                    RATIO(num100, tot), RATIO(num75, tot), RATIO(num50, tot), RATIO(num25, tot), RATIO(less25, tot));
            }
        }
        printf("in_max %d\n", in_max[0]);
    }
};

struct counter {
    unsigned long long total;
    unsigned long long* value;

    __host__ counter() {}
    __host__ counter(unsigned long long tot) :total(tot) {}

    __host__ void allocateDeviceMem() {
        checkCudaErrors(cudaMalloc((void**)& value, sizeof(unsigned long long)));
    }

    __host__ void freeDeviceMem() const {
        checkCudaErrors(cudaFree(value));
    }

    __device__ void reset() {
        value[0] = 0;
    }

    __device__ void increment(int val) {
        atomicAdd(value, val);
    }

    __device__ void print(int iteration, bool last) const {
        //if (!last) return;

        unsigned long long val = value[0];
        printf("iteration %4d: total %7llu, value %7llu %6.2f%%\n", iteration, total, val, RATIO(val, total));
    }
};

// counter that can handle multiple inner iterations
struct MultiIterCounter {
    int print_out_iter;
    int max_in_iter;

    unsigned long long* values;
    unsigned long long* in_iter;
    int* out_iter; // outer iteration computed by this metric
    int* in_max; // max in_iter encountered even if not recorded

    __host__ MultiIterCounter() {}
    __host__ MultiIterCounter(int _print_out_iter, int _max_in_iter) : print_out_iter(_print_out_iter), max_in_iter(_max_in_iter) {}

    __host__ void allocateDeviceMem() {
        checkCudaErrors(cudaMalloc((void**)& values, max_in_iter * sizeof(unsigned long long)));
        checkCudaErrors(cudaMalloc((void**)& in_iter, sizeof(unsigned long long)));
        checkCudaErrors(cudaMalloc((void**)& out_iter, sizeof(int)));
        checkCudaErrors(cudaMalloc((void**)& in_max, sizeof(int)));
    }

    __host__ void freeDeviceMem() const {
        checkCudaErrors(cudaFree(values));
        checkCudaErrors(cudaFree(in_iter));
        checkCudaErrors(cudaFree(out_iter));
        checkCudaErrors(cudaFree(in_max));
    }

    __device__ void reset(int pid, bool first) {
        if (first) {
            if (pid < max_in_iter)
                values[pid] = 0;
            in_max[0] = 0;
            out_iter[0] = 0;
            in_iter[0] = 0;
        }
        if (pid == 0)
            out_iter[0]++;
    }

    __device__ void increment(int lane_id, int in_it) {
        if (out_iter[0] != print_out_iter)
            return;

        atomicMax(in_max, in_it);

        if (in_it < max_in_iter) {
            // first active thread of the warp should increment the metrics
            const int num_active = __popc(__activemask());
            const int idx_lane = __popc(__activemask() & ((1u << lane_id) - 1));
            if (idx_lane == 0)
                atomicAdd(values + in_it, num_active);
            atomicMax(in_iter, in_it);
        }
    }

    __device__ void print(bool last) const {
        if (out_iter[0] == print_out_iter) {
            for (size_t i = 0; i < in_iter[0]; i += 40) {
                printf("it: %5d ", i);
                for (int j = 0; j < 40 && (i + j) < in_iter[0]; j++)
                    printf("%4llu ", values[i + j]);
                printf("\n");
            }
            printf("in_max %d\n", in_max[0]);
        }
    }
};

struct HistoCounter {
    int min;
    int max;
    int numBins;
    int binWidth;

    unsigned long long* bins;

    __host__ HistoCounter() {}
    __host__ HistoCounter(int _min, int _max, int _numBines) :min(_min), max(_max), numBins(_numBines + 2), binWidth((_max - _min) / _numBines) {}

    __host__ void allocateDeviceMem() {
        checkCudaErrors(cudaMalloc((void**)& bins, (numBins + 2) * sizeof(unsigned long long))); // + < min and >= max
    }

    __host__ void freeDeviceMem() const {
        checkCudaErrors(cudaFree(bins));
    }

    __device__ void reset(int pid, bool first) {
        if (pid < numBins)
            bins[pid] = 0;
    }

    __device__ void increment(int value) {
        // compute bin corresponding to value
        int binId;
        if (value < min)
            binId = 0;
        else if (value >= max)
            binId = numBins - 1;
        else // min <= value < max
            binId = (value - min) / binWidth + 1; // +1 because bin 0 if for value < min

        atomicAdd(bins + binId, 1);
    }

    __device__ void print(int iteration, float elapsedSeconds) const {
        // sum all bins, so we can compute percentiles
        unsigned long long total = 0;
        for (size_t i = 0; i < numBins; i++)
            total += bins[i];
        if (total == 0)
            return; // nothing to print
        printf("iter %4d,tot %5llu,<%d:%6.2f%%,", iteration, total, min, RATIO(bins[0], total));
        int left = min;
        for (size_t i = 1; i < numBins - 1; i++, left += binWidth)
            printf("<%d:%6.2f%%,", left + binWidth, RATIO(bins[i], total));
        printf(">=%d:%6.2f%%\n", max, RATIO(bins[numBins - 1], total));
    }
};

struct lanes_histo {
    unsigned long long* total;
    unsigned long long* by25;
    unsigned long long* by50;
    unsigned long long* by75;
    unsigned long long* by100;

    __host__ void allocateDeviceMem() {
        checkCudaErrors(cudaMalloc((void**)& total, sizeof(unsigned long long)));
        checkCudaErrors(cudaMalloc((void**)& by25, sizeof(unsigned long long)));
        checkCudaErrors(cudaMalloc((void**)& by50, sizeof(unsigned long long)));
        checkCudaErrors(cudaMalloc((void**)& by75, sizeof(unsigned long long)));
        checkCudaErrors(cudaMalloc((void**)& by100, sizeof(unsigned long long)));
    }

    __host__ void freeDeviceMem() const {
        checkCudaErrors(cudaFree(total));
        checkCudaErrors(cudaFree(by25));
        checkCudaErrors(cudaFree(by50));
        checkCudaErrors(cudaFree(by75));
        checkCudaErrors(cudaFree(by100));
    }

    __device__ void reset() {
        total[0] = 0;
        by25[0] = 0;
        by50[0] = 0;
        by75[0] = 0;
        by100[0] = 0;
    }

    __device__ void increment(int lane_id) {
        // first active thread of the warp should increment the metrics
        const int num_active = __popc(__activemask());
        const int idx_lane = __popc(__activemask() & ((1u << lane_id) - 1));
        if (idx_lane == 0) {
            atomicAdd(total, 1);
            if (num_active == 32)
                atomicAdd(by100, 1);
            else if (num_active >= 24)
                atomicAdd(by75, 1);
            else if (num_active >= 16)
                atomicAdd(by50, 1);
            else if (num_active >= 8)
                atomicAdd(by25, 1);
        }
    }

    __device__ void print(int iteration, float elapsedSeconds) const {
        unsigned long long tot = total[0];
        if (tot > 0) {
            unsigned long long num100 = by100[0];
            unsigned long long num75 = by75[0];
            unsigned long long num50 = by50[0];
            unsigned long long num25 = by25[0];
            unsigned long long less25 = tot - num100 - num75 - num50 - num25;
            printf("iter %4d: elapsed %.2fs, total %7llu, 100%% %6.2f%%, >=75%% %6.2f%%, >=50%% %6.2f%%, >=25%% %6.2f%%, less %6.2f%%\n",
                iteration, elapsedSeconds, tot, RATIO(num100, tot), RATIO(num75, tot), RATIO(num50, tot), RATIO(num25, tot), RATIO(less25, tot));
        }
    }
};

struct metrics {
    unsigned int* num_active_paths;
    lanes_histo lanes_cnt;
    counter cnt;
    multi_iter_warp_counter multi;
    HistoCounter histo;
    MultiIterCounter multiIterCounter;

    __host__ metrics() {
        multi = multi_iter_warp_counter(100, 73);
        histo = HistoCounter(8000, 10000, 10);
        multiIterCounter = MultiIterCounter(73, 100);
        cnt = counter(1024 * 1024);
    }

    __host__ void allocateDeviceMem() {
        lanes_cnt.allocateDeviceMem();
        cnt.allocateDeviceMem();
        multi.allocateDeviceMem();
        histo.allocateDeviceMem();
        multiIterCounter.allocateDeviceMem();
    }

    __host__ void freeDeviceMem() const {
        lanes_cnt.freeDeviceMem();
        cnt.freeDeviceMem();
        multi.freeDeviceMem();
        histo.freeDeviceMem();
        multiIterCounter.freeDeviceMem();
    }

    __device__ void reset(int pid, bool first) {
        if (pid == 0) {
            num_active_paths[0] = 0;
            lanes_cnt.reset();
            //if (first)
            cnt.reset();
        }
        multi.reset(pid, first);
        histo.reset(pid, first);
        multiIterCounter.reset(pid, first);
    }

    __device__ void print(int iteration, float elapsedSeconds, bool last) const {
        lanes_cnt.print(iteration, elapsedSeconds);
        //cnt.print(iteration, last);
        //multi.print();
        //histo.print(iteration, elapsedSeconds);
        //multiIterCounter.print(last);
    }
};
