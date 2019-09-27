#include <float.h>
#include <cuda_profiler_api.h>

#include "ray.h"
#include "camera.h"
#include "scene.h"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define DYNAMIC_FETCH_THRESHOLD 20          // If fewer than this active, fetch new rays

const int MaxBlockWidth = 32;
const int MaxBlockHeight = 2; // block width is 32
const int kMaxBounces = 10;

typedef unsigned long long ull;

struct render_params {
    vec3* fb;
    scene sc;
    unsigned int width;
    unsigned int height;
    unsigned int spp;
    unsigned int maxActivePaths;
    ull samples_count;
};

#define RATIO(x,a)  (100.0 * x / a)

struct multi_iter_warp_counter {
    int print_out_iter;
    int max_in_iter;

    int *out_iter;
    int *in_iter;

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
    __host__ MultiIterCounter(int _print_out_iter, int _max_in_iter): print_out_iter(_print_out_iter), max_in_iter(_max_in_iter) {}

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
        cnt = counter(1024*1024);
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

struct paths {
    ull* next_sample; // used by init() to track next sample to fetch

    // pixel_id of active paths currently being traced by the renderer, it's a subset of all_sample_pool
    unsigned int* active_paths;
    unsigned int* next_path; // used by hit_bvh() to track next path to fetch and trace

    ray* r;
    ray* shadow;
    rand_state* state;
    vec3* attentuation;
    vec3* emitted;
    unsigned short* flag;
    int* hit_id;
    vec3* hit_normal;
    float* hit_t;

    metrics m;
};

#define FLAG_BOUNCE_MASK    0xF
#define FLAG_HAS_HIT        0x10
#define FLAG_HAS_SHADOW     0x20
#define FLAG_SHADOW_HIT     0x40

void setup_paths(paths& p, int nx, int ny, int ns, unsigned int maxActivePaths) {
    // at any given moment only kMaxActivePaths at most are active at the same time
    const unsigned num_paths = maxActivePaths;
    checkCudaErrors(cudaMalloc((void**)& p.r, num_paths * sizeof(ray)));
    checkCudaErrors(cudaMalloc((void**)& p.shadow, num_paths * sizeof(ray)));
    checkCudaErrors(cudaMalloc((void**)& p.state, num_paths * sizeof(rand_state)));
    checkCudaErrors(cudaMalloc((void**)& p.attentuation, num_paths * sizeof(vec3)));
    checkCudaErrors(cudaMalloc((void**)& p.emitted, num_paths * sizeof(vec3)));
    checkCudaErrors(cudaMalloc((void**)& p.flag, num_paths * sizeof(unsigned short)));
    checkCudaErrors(cudaMalloc((void**)& p.hit_id, num_paths * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)& p.hit_normal, num_paths * sizeof(vec3)));
    checkCudaErrors(cudaMalloc((void**)& p.hit_t, num_paths * sizeof(float)));

    checkCudaErrors(cudaMalloc((void**)& p.active_paths, num_paths * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**)& p.next_path, sizeof(unsigned int)));

    checkCudaErrors(cudaMalloc((void**)& p.next_sample, sizeof(ull)));
    checkCudaErrors(cudaMemset((void*)p.next_sample, 0, sizeof(ull)));
    checkCudaErrors(cudaMalloc((void**)& p.m.num_active_paths, sizeof(unsigned int)));
    p.m.allocateDeviceMem();
}

void free_paths(const paths& p) {
    checkCudaErrors(cudaFree(p.r));
    checkCudaErrors(cudaFree(p.shadow));
    checkCudaErrors(cudaFree(p.state));
    checkCudaErrors(cudaFree(p.attentuation));
    checkCudaErrors(cudaFree(p.emitted));
    checkCudaErrors(cudaFree(p.flag));
    checkCudaErrors(cudaFree(p.hit_id));
    checkCudaErrors(cudaFree(p.hit_normal));
    checkCudaErrors(cudaFree(p.hit_t));
    checkCudaErrors(cudaFree(p.next_sample));

    checkCudaErrors(cudaFree(p.active_paths));
    checkCudaErrors(cudaFree(p.next_path));

    checkCudaErrors(cudaFree(p.m.num_active_paths));
    p.m.freeDeviceMem();
}

__global__ void init(const render_params params, paths p, bool first, const camera cam) {
    // kMaxActivePaths threads are started to fetch the samples from all_sample_pool and initialize the paths
    // to keep things simple a block contains a single warp so that we only need to keep a single shared nextSample per block

    const unsigned int pid = threadIdx.x + blockIdx.x * blockDim.x;
    if (pid == 0)
        p.next_path[0] = 0;
    p.m.reset(pid, first);
    __syncthreads();

    if (pid >= params.maxActivePaths)
        return;

    rand_state state;
    unsigned short bounce;
    if (first) {
        // this is the very first init, all paths are marked terminated, and we don't have a valid random state yet
        state = (wang_hash(pid) * 336343633) | 1;
        bounce = kMaxBounces;
    } else {
        state = p.state[pid];
        bounce = p.flag[pid] & FLAG_BOUNCE_MASK;
    }

    // generate all terminated paths
    const bool          terminated     = bounce == kMaxBounces;
    const unsigned int  maskTerminated = __ballot_sync(__activemask(), terminated);
    const int           numTerminated  = __popc(maskTerminated);
    const int           idxTerminated  = __popc(maskTerminated & ((1u << threadIdx.x) - 1));

    __shared__ volatile ull nextSample;

    if (terminated) {
        // first terminated lane increments next_sample
        if (idxTerminated == 0)
            nextSample = atomicAdd(p.next_sample, numTerminated);

        // compute sample this lane is going to fetch
        const ull sample_id = nextSample + idxTerminated;
        if (sample_id >= params.samples_count)
            return; // no more samples to fetch

        // retrieve pixel_id corresponding to current path
        const unsigned int pixel_id = sample_id / params.spp;
        p.active_paths[pid] = pixel_id;

        // compute pixel coordinates
        const unsigned int x = pixel_id % params.width;
        const unsigned int y = pixel_id / params.width;

        // generate camera ray
        float u = float(x + random_float(state)) / float(params.width);
        float v = float(y + random_float(state)) / float(params.height);
        p.r[pid] = cam.get_ray(u, v, state);
        p.state[pid] = state;
        p.attentuation[pid] = vec3(1, 1, 1);
        p.flag[pid] = 0;
    }

    // path still active or has just been generated
    //TODO each warp uses activemask() to count number active lanes then just 1st active lane need to update the metric
    atomicAdd(p.m.num_active_paths, 1);
}

#define IDX_SENTINEL    0
#define IS_DONE(idx)    (idx == IDX_SENTINEL)
#define IS_LEAF(idx)    (idx >= sc.count)

#define BIT_DONE        3
#define BIT_MASK        3
#define BIT_PARENT      0
#define BIT_LEFT        1
#define BIT_RIGHT       2

__device__ void pop_bitstack(unsigned long long& bitstack, int& idx) {
    const int m = (__ffsll(bitstack) - 1) / 2;
    bitstack >>= (m << 1);
    idx >>= m;

    if (bitstack == BIT_DONE) {
        idx = IDX_SENTINEL;
    }
    else {
        // idx could point to left or right child regardless of sibling we need to go to
        idx = (idx >> 1) << 1; // make sure idx always points to left sibling
        idx += (bitstack & BIT_MASK) - 1; // move idx to the sibling stored in bitstack
        bitstack = bitstack & (~BIT_MASK); // set bitstack to parent, so we can backtrack
    }
}

__global__ void trace_scattered(const render_params params, paths p) {
    // a limited number of threads are started to operate on active_paths

    unsigned int pid = 0; // currently traced path
    ray r; // corresponding ray

    // bvh traversal state
    int idx = IDX_SENTINEL;
    bool found;
    float closest;
    hit_record rec;

    unsigned long long bitstack;

    // Initialize persistent threads.
    // given that each block is 32 thread wide, we can use threadIdx.x as a warpId
    __shared__ volatile int nextPathArray[MaxBlockHeight]; // Current ray index in global buffer.
    __shared__ volatile bool noMorePaths[MaxBlockHeight]; // true when no more paths are available to fetch

    // Persistent threads: fetch and process rays in a loop.

    while (true) {
        const int tidx = threadIdx.x;
        volatile int& pathBase = nextPathArray[threadIdx.y];
        volatile bool& noMoreP = noMorePaths[threadIdx.y];

        // identify which lanes are done
        const bool          terminated      = IS_DONE(idx);
        const unsigned int  maskTerminated  = __ballot_sync(__activemask(), terminated);
        const int           numTerminated   = __popc(maskTerminated);
        const int           idxTerminated   = __popc(maskTerminated & ((1u << tidx) - 1));

        if (terminated) {
            // first terminated lane updates the base ray index
            if (idxTerminated == 0) {
                pathBase = atomicAdd(p.next_path, numTerminated);
                noMoreP = (pathBase + numTerminated) >= params.maxActivePaths;
            }

            pid = pathBase + idxTerminated;
            if (pid >= params.maxActivePaths) {
                return;
            }

            found = false; // always reset found to avoid writing hit information for terminated paths
            // setup ray if path not already terminated
            if ((p.flag[pid] & FLAG_BOUNCE_MASK) < kMaxBounces) {
                // Fetch ray
                r = p.r[pid];

                // idx is already set to IDX_SENTINEL, but make sure we set found to false
                idx = 1;
                closest = FLT_MAX;
                bitstack = BIT_DONE;
            }
        }

        //if (__popc(__activemask()) < 16) {
        //    // just mark the path as no hit
        //    idx = IDX_SENTINEL;
        //}

        // traversal
        const scene& sc = params.sc;
        while (!IS_DONE(idx)) {
            //p.m.lanes_cnt.increment(tidx);

            // we already intersected ray with idx node, now we need to load its children and intersect the ray with them
            if (!IS_LEAF(idx)) {
                // load left, right nodes
                bvh_node left, right;
                const int idx2 = idx * 2; // we are going to load and intersect children of idx
                if (idx2 < 2048) {
                    left = d_nodes[idx2];
                    right = d_nodes[idx2 + 1];
                }
                else {
                    // each spot in the texture holds two children, that's why we devide the relative texture index by 2
                    unsigned int tex_idx = ((idx2 - 2048) >> 1) * 3;
                    float4 a = tex1Dfetch(t_bvh, tex_idx++);
                    float4 b = tex1Dfetch(t_bvh, tex_idx++);
                    float4 c = tex1Dfetch(t_bvh, tex_idx++);
                    left = bvh_node(a.x, a.y, a.z, a.w, b.x, b.y);
                    right = bvh_node(b.z, b.w, c.x, c.y, c.z, c.w);
                }

                const float left_t = hit_bbox(left, r, closest);
                const bool traverse_left = left_t < FLT_MAX;
                const float right_t = hit_bbox(right, r, closest);
                const bool traverse_right = right_t < FLT_MAX;

                const bool swap = right_t < left_t; // right child is closer

                if (traverse_left || traverse_right) {
                    idx = idx2 + swap; // intersect closer node next
                    if (traverse_left && traverse_right) // push farther node into the stack
                        bitstack = (bitstack << 2) + (swap ? BIT_LEFT : BIT_RIGHT);
                    else // push parent bit to the stack to backtrack later
                        bitstack = (bitstack << 2) + BIT_PARENT;
                }
                else {
                    pop_bitstack(bitstack, idx);
                }
            } else {
                int m = (idx - sc.count) * lane_size_float;
                #pragma unroll
                for (int i = 0; i < lane_size_spheres; i++) {
                    float x = tex1Dfetch(t_spheres, m++);
                    float y = tex1Dfetch(t_spheres, m++);
                    float z = tex1Dfetch(t_spheres, m++);
                    vec3 center(x, y, z);
                    if (hit_point(center, r, 0.001f, closest, rec)) {
                        found = true;
                        closest = rec.t;
                        rec.idx = (idx - sc.count) * lane_size_spheres + i;
                    }
                }

                if (found) // exit traversal once we find an intersection in any leaf
                    idx = IDX_SENTINEL;
                else
                    pop_bitstack(bitstack, idx);
            }

            // some lanes may have already exited the loop, if not enough active thread are left, exit the loop
            if (!noMoreP && __popc(__activemask()) < DYNAMIC_FETCH_THRESHOLD)
                break;
        }

        if (found && IS_DONE(idx)) {
            // finished traversing bvh
            p.hit_id[pid] = rec.idx;
            p.hit_normal[pid] = rec.n;
            p.hit_t[pid] = rec.t;
            p.flag[pid] = p.flag[pid] | FLAG_HAS_HIT;
        }
    }
}

// generate shadow rays for all non terminated rays with intersections
__global__ void generate_shadow_raws(const render_params params, paths p) {

    const vec3 light_center(5000, 0, 0);
    const float light_radius = 500;
    const float light_emissive = 100;

    // kMaxActivePaths threads update all p.num_active_paths
    const unsigned int pid = threadIdx.x + blockIdx.x * blockDim.x;
    if (pid == 0)
        p.next_path[0] = 0;
    __syncthreads();

    if (pid >= params.maxActivePaths)
        return;

    // if the path has no intersection, which includes terminated paths, do nothing
    const unsigned short flag = p.flag[pid];
    if (!(flag & FLAG_HAS_HIT))
        return;

    const ray r = p.r[pid];
    const float hit_t = p.hit_t[pid];
    const vec3 hit_p = r.point_at_parameter(hit_t);
    const vec3 hit_n = p.hit_normal[pid];
    rand_state state = p.state[pid];

    // create a random direction towards the light
    // coord system for sampling
    const vec3 sw = unit_vector(light_center - hit_p);
    const vec3 su = unit_vector(cross(fabs(sw.x()) > 0.01f ? vec3(0, 1, 0) : vec3(1, 0, 0), sw));
    const vec3 sv = cross(sw, su);

    // sample sphere by solid angle
    const float cosAMax = sqrt(1.0f - light_radius * light_radius / (hit_p - light_center).squared_length());
    const float eps1 = random_float(state);
    const float eps2 = random_float(state);
    const float cosA = 1.0f - eps1 + eps1 * cosAMax;
    const float sinA = sqrt(1.0f - cosA * cosA);
    const float phi = 2 * kPI * eps2;
    const vec3 l = unit_vector(su * cosf(phi) * sinA + sv * sinf(phi) * sinA + sw * cosA);

    p.state[pid] = state;
    const float dotl = dot(l, hit_n);
    if (dotl <= 0)
        return;

    const float omega = 2 * kPI * (1.0f - cosAMax);
    p.shadow[pid] = ray(hit_p, l);
    p.emitted[pid] = vec3(light_emissive, light_emissive, light_emissive) * dotl * omega / kPI;
    p.flag[pid] = flag | FLAG_HAS_SHADOW;
}

// traces all paths that have FLAG_HAS_SHADOW set, sets FLAG_SHADOW_HIT to true if there is a hit
__global__ void trace_shadows(const render_params params, paths p) {
    // a limited number of threads are started to operate on active_paths

    unsigned int pid = 0; // currently traced path
    ray r; // corresponding ray

    // bvh traversal state
    int idx = IDX_SENTINEL;
    bool found = false;
    hit_record rec;

    unsigned long long bitstack;

    // Initialize persistent threads.
    // given that each block is 32 thread wide, we can use threadIdx.x as a warpId
    __shared__ volatile int nextPathArray[MaxBlockHeight]; // Current ray index in global buffer.

    // Persistent threads: fetch and process rays in a loop.

    while (true) {
        const int tidx = threadIdx.x;
        volatile int& pathBase = nextPathArray[threadIdx.y];

        // identify which lanes are done
        const bool          terminated = IS_DONE(idx);
        const unsigned int  maskTerminated = __ballot_sync(__activemask(), terminated);
        const int           numTerminated = __popc(maskTerminated);
        const int           idxTerminated = __popc(maskTerminated & ((1u << tidx) - 1));

        if (terminated) {
            // first terminated lane updates the base ray index
            if (idxTerminated == 0)
                pathBase = atomicAdd(p.next_path, numTerminated);

            pid = pathBase + idxTerminated;
            if (pid >= params.maxActivePaths)
                return;

            // setup ray if path has a shadow ray
            if ((p.flag[pid] & FLAG_HAS_SHADOW)) {
                // Fetch ray
                r = p.shadow[pid];

                // idx is already set to IDX_SENTINEL, but make sure we set found to false
                found = false;
                idx = 1;
                bitstack = BIT_DONE;
            }
        }

        // traversal
        const scene& sc = params.sc;
        while (!IS_DONE(idx)) {
            // we already intersected ray with idx node, now we need to load its children and intersect the ray with them
            if (!IS_LEAF(idx)) {
                // load left, right nodes
                bvh_node left, right;
                const int idx2 = idx * 2; // we are going to load and intersect children of idx
                if (idx2 < 2048) {
                    left = d_nodes[idx2];
                    right = d_nodes[idx2 + 1];
                }
                else {
                    // each spot in the texture holds two children, that's why we devide the relative texture index by 2
                    unsigned int tex_idx = ((idx2 - 2048) >> 1) * 3;
                    float4 a = tex1Dfetch(t_bvh, tex_idx++);
                    float4 b = tex1Dfetch(t_bvh, tex_idx++);
                    float4 c = tex1Dfetch(t_bvh, tex_idx++);
                    left = bvh_node(a.x, a.y, a.z, a.w, b.x, b.y);
                    right = bvh_node(b.z, b.w, c.x, c.y, c.z, c.w);
                }

                const float left_t = hit_bbox(left, r, FLT_MAX);
                const bool traverse_left = left_t < FLT_MAX;
                const float right_t = hit_bbox(right, r, FLT_MAX);
                const bool traverse_right = right_t < FLT_MAX;

                const bool swap = right_t < left_t; // right child is closer

                if (traverse_left || traverse_right) {
                    idx = idx2 + swap; // intersect closer node next
                    if (traverse_left && traverse_right) // push farther node into the stack
                        bitstack = (bitstack << 2) + (swap ? BIT_LEFT : BIT_RIGHT);
                    else // push parent bit to the stack to backtrack later
                        bitstack = (bitstack << 2) + BIT_PARENT;
                }
                else {
                    pop_bitstack(bitstack, idx);
                }
            }
            else {
                int m = (idx - sc.count) * lane_size_float;
                #pragma unroll
                for (int i = 0; i < lane_size_spheres && !found; i++) {
                    float x = tex1Dfetch(t_spheres, m++);
                    float y = tex1Dfetch(t_spheres, m++);
                    float z = tex1Dfetch(t_spheres, m++);
                    vec3 center(x, y, z);
                    found = hit_point(center, r, 0.001f, FLT_MAX, rec);
                }

                if (found) // exit traversal once we find an intersection in any leaf
                    idx = IDX_SENTINEL;
                else
                    pop_bitstack(bitstack, idx);
            }

            // some lanes may have already exited the loop, if not enough active thread are left, exit the loop
            if (__popc(__activemask()) < DYNAMIC_FETCH_THRESHOLD) {
                break;
            }
        }

        if (found) {
            // finished traversing bvh
            p.flag[pid] = p.flag[pid] | FLAG_SHADOW_HIT;
        }
    }
}

// for all non terminated rays, accounts for shadow hit, compute scattered ray and resets the flag
__global__ void update(const render_params params, paths p) {
    const float sky_emissive = .2f;

    // kMaxActivePaths threads update all p.num_active_paths
    const unsigned int pid = threadIdx.x + blockIdx.x * blockDim.x;
    if (pid >= params.maxActivePaths)
        return;

    // is the path already done ?
    unsigned short flag = p.flag[pid];
    unsigned short bounce = flag & FLAG_BOUNCE_MASK;
    if (bounce == kMaxBounces)
        return; // yup, done and already taken care of

    // did the ray hit a primitive ?
    if (flag & FLAG_HAS_HIT) {
        // update path attenuation
        const int hit_id = p.hit_id[pid];
        int clr_idx = params.sc.colors[hit_id] * 3;
        const vec3 albedo = vec3(d_colormap[clr_idx++], d_colormap[clr_idx++], d_colormap[clr_idx++]);
        
        vec3 attenuation = p.attentuation[pid] * albedo;
        p.attentuation[pid] = attenuation;

        // scatter ray, only if we didn't reach kMaxBounces
        bounce++;
        if (bounce < kMaxBounces) {
            const ray r = p.r[pid];
            const float hit_t = p.hit_t[pid];
            const vec3 hit_p = r.point_at_parameter(hit_t);

            const vec3 hit_n = p.hit_normal[pid];
            rand_state state = p.state[pid];
            const vec3 target = hit_n + random_in_unit_sphere(state);

            p.r[pid] = ray(hit_p, target);
            p.state[pid] = state;
        }

        // account for light contribution if no shadow hit
        if ((flag & FLAG_HAS_SHADOW) && !(flag & FLAG_SHADOW_HIT)) {
            const vec3 incoming = p.emitted[pid] * attenuation;
            const unsigned int pixel_id = p.active_paths[pid];
            atomicAdd(params.fb[pixel_id].e, incoming.e[0]);
            atomicAdd(params.fb[pixel_id].e + 1, incoming.e[1]);
            atomicAdd(params.fb[pixel_id].e + 2, incoming.e[2]);
        }
    }
    else {
        if (bounce > 0) {
            const vec3 incoming = p.attentuation[pid] * sky_emissive;
            const unsigned int pixel_id = p.active_paths[pid];
            atomicAdd(params.fb[pixel_id].e, incoming.e[0]);
            atomicAdd(params.fb[pixel_id].e + 1, incoming.e[1]);
            atomicAdd(params.fb[pixel_id].e + 2, incoming.e[2]);
        }
        bounce = kMaxBounces; // mark path as terminated
    }

    p.flag[pid] = bounce;
}

__global__ void print_metrics(metrics m, unsigned int iteration, unsigned int maxActivePaths, float elapsedSeconds, bool last) {
    m.print(iteration, elapsedSeconds, last);
}

camera setup_camera(int nx, int ny, float dist) {
    vec3 lookfrom(dist, dist, dist);
    vec3 lookat(0, 0, 0);
    float dist_to_focus = (lookfrom - lookat).length();
    float aperture = 0.1;
    return camera(lookfrom,
        lookat,
        vec3(0, 1, 0),
        30.0,
        float(nx) / float(ny),
        aperture,
        dist_to_focus);
}

// http://chilliant.blogspot.com.au/2012/08/srgb-approximations-for-hlsl.html
static uint32_t LinearToSRGB(float x)
{
    x = max(x, 0.0f);
    x = max(1.055f * powf(x, 0.416666667f) - 0.055f, 0.0f);
    uint32_t u = min((uint32_t)(x * 255.9f), 255u);
    return u;
}

void write_image(const char* output_file, const vec3 *fb, const int nx, const int ny, const int ns) {
    char *data = new char[nx * ny * 3];
    int idx = 0;
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            data[idx++] = LinearToSRGB(fb[pixel_index].r() / ns);
            data[idx++] = LinearToSRGB(fb[pixel_index].g() / ns);
            data[idx++] = LinearToSRGB(fb[pixel_index].b() / ns);
        }
    }
    stbi_write_png(output_file, nx, ny, 3, (void*)data, nx * 3);
    delete[] data;
}

int cmpfunc(const void * a, const void * b) {
    if (*(double*)a > *(double*)b)
        return 1;
    else if (*(double*)a < *(double*)b)
        return -1;
    else
        return 0;
}

#define ARG_INT(idx, def)   (argc > idx ? strtol(argv[idx], NULL, 10) : def)

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "usage spheres file_name [width=1200] [height=1200] [num_samples=1] [camera_dist=100] [maxActivePaths=1M] [numBouncesPerIter=4] [colormap=viridis.csv] [verbose]";
        exit(-1);
    }
    char* input = argv[1];
    const int nx = ARG_INT(2, 1200);
    const int ny = ARG_INT(3, 1200);
    const int ns = ARG_INT(4, 10);
    const int dist = ARG_INT(5, 100);
    const int maxActivePaths = ARG_INT(6, 1024 * 1024);
    const int numBouncesPerIter = ARG_INT(7, 4);
    const char* colormap = (argc > 8) ? argv[8] : "viridis.csv";
    const bool verbose = (argc > 9 && !strcmp(argv[9], "verbose"));

    const bool is_csv = strncmp(input + strlen(input) - 4, ".csv", 4) == 0;
    
    cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel, maxActivePaths = " << maxActivePaths << ", numBouncesPerIter = " << numBouncesPerIter << "\n";

    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    // allocate FB
    vec3 *d_fb;
    checkCudaErrors(cudaMalloc((void **)&d_fb, fb_size));
    checkCudaErrors(cudaMemset(d_fb, 0, fb_size));


    // load colormap
    vector<vector<float>> data = parse2DCsvFile(colormap);
    std::cout << "colormap contains " << data.size() << " points\n";
    float *_viridis_data = new float[data.size() * 3];
    int idx = 0;
    for (auto l : data) {
        _viridis_data[idx++] = (float)l[0];
        _viridis_data[idx++] = (float)l[1];
        _viridis_data[idx++] = (float)l[2];
    }
    // setup scene
    scene sc;
    setup_scene(input, sc, is_csv, _viridis_data);
    delete[] _viridis_data;
    _viridis_data = NULL;

    camera cam = setup_camera(nx, ny, dist);
    vec3* h_fb = new vec3[fb_size];

    render_params params;
    params.fb = d_fb;
    params.sc = sc;
    params.width = nx;
    params.height = ny;
    params.spp = ns;
    params.maxActivePaths = maxActivePaths;
    params.samples_count = nx;
    params.samples_count *= ny;
    params.samples_count *= ns;

    paths p;
    setup_paths(p, nx, ny, ns, maxActivePaths);

    cout << "started renderer\n" << std::flush;
    clock_t start = clock();
    cudaProfilerStart();

    unsigned int iteration = 0;
    while (true) {

        // init kMaxActivePaths using equal number of threads
        {
            const int threads = 32; // 1 warp per block
            const int blocks = (maxActivePaths + threads - 1) / threads;
            init << <blocks, threads >> > (params, p, iteration == 0, cam);
            checkCudaErrors(cudaGetLastError());
        }

        // check if not all paths terminated
        // we don't want to check the metric after each bounce, we do it every numBouncesPerIter iterations
        if (iteration > 0 && (iteration % numBouncesPerIter) == 0) {
            unsigned int num_active_paths;
            checkCudaErrors(cudaMemcpy((void*)& num_active_paths, (void*)p.m.num_active_paths, sizeof(unsigned int), cudaMemcpyDeviceToHost));
            if (num_active_paths < (maxActivePaths * 0.05f)) {
                break;
            }
        }

        // traverse bvh
        {
            dim3 blocks(6400 * 2, 1);
            dim3 threads(MaxBlockWidth, MaxBlockHeight);
            trace_scattered << <blocks, threads >> > (params, p);
            checkCudaErrors(cudaGetLastError());
        }

        // generate shadow rays
        {
            const int threads = 128;
            const int blocks = (maxActivePaths + threads - 1) / threads;
            generate_shadow_raws << <blocks, threads >> > (params, p);
            checkCudaErrors(cudaGetLastError());
        }

        // trace shadow rays
        {
            dim3 blocks(6400 * 2, 1);
            dim3 threads(MaxBlockWidth, MaxBlockHeight);
            trace_shadows << <blocks, threads >> > (params, p);
            checkCudaErrors(cudaGetLastError());
        }

        // update paths accounting for intersection and light contribution
        {
            const int threads = 128;
            const int blocks = (maxActivePaths + threads - 1) / threads;
            update << <blocks, threads >> > (params, p);
            checkCudaErrors(cudaGetLastError());
        }

        // print metrics
        if (verbose) {
            print_metrics << <1, 1 >> > (p.m, iteration, maxActivePaths, (float)(clock() - start) / CLOCKS_PER_SEC, false);
            checkCudaErrors(cudaGetLastError());
        }
        //checkCudaErrors(cudaDeviceSynchronize());

        iteration++;
    }
    cudaProfilerStop();

    print_metrics << <1, 1 >> > (p.m, iteration, maxActivePaths, (float)(clock() - start) / CLOCKS_PER_SEC, true);
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaDeviceSynchronize());
    cerr << "\rrendered " << params.samples_count << " samples in " << (float)(clock() - start) / CLOCKS_PER_SEC << " seconds.                                    \n";

    // Output FB as Image
    checkCudaErrors(cudaMemcpy(h_fb, d_fb, fb_size, cudaMemcpyDeviceToHost));
    char file_name[100];
    sprintf(file_name, "%s_%dx%dx%d_%d_bvh.png", input, nx, ny, ns, dist);
    write_image(file_name, h_fb, nx, ny, ns);
    delete[] h_fb;
    h_fb = NULL;

    // clean up
    free_paths(p);
    releaseScene(sc);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(params.fb));

    cudaDeviceReset();
}