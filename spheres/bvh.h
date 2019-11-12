#pragma once

/**
non-leaf nodes represent bounding box of all spheres that are inside it
leaf nodes represent 2 spheres
*/
struct bvh_node {
    __host__ __device__ bvh_node() {}
    bvh_node(const vec3& A, const vec3& B) :a(A), b(B) {}
    __device__ bvh_node(float x0, float y0, float z0, float x1, float y1, float z1) : a(x0, y0, z0), b(x1, y1, z1) {}

    __device__ vec3 min() const { return a; }
    __device__ vec3 max() const { return b; }

    unsigned int split_axis() const { return max_component(b - a); }

    vec3 a;
    vec3 b;
};

std::ostream& operator << (std::ostream& out, const bvh_node& node) {
    out << "{" << node.a << ", " << node.b << "}";
    return out;
}

int box_x_compare(const void* a, const void* b) {
    float xa = ((sphere*)a)->center.x();
    float xb = ((sphere*)b)->center.x();

    if (xa < xb) return -1;
    else if (xb < xa) return 1;
    return 0;
}

int box_y_compare(const void* a, const void* b) {
    float ya = ((sphere*)a)->center.y();
    float yb = ((sphere*)b)->center.y();

    if (ya < yb) return -1;
    else if (yb < ya) return 1;
    return 0;
}

int box_z_compare(const void* a, const void* b) {
    float za = ((sphere*)a)->center.z();
    float zb = ((sphere*)b)->center.z();

    if (za < zb) return -1;
    else if (zb < za) return 1;
    return 0;
}

vec3 minof(const sphere* l, int n) {
    vec3 min = l[0].center;
    for (int i = 1; i < n; i++) {
        for (int a = 0; a < 3; a++)
            min[a] = fminf(min[a], l[i].center[a]);
    }
    return min;
}

vec3 maxof(const sphere* l, int n) {
    vec3 max = l[0].center;
    for (int i = 1; i < n; i++) {
        for (int a = 0; a < 3; a++)
            max[a] = fmaxf(max[a], l[i].center[a]);
    }
    return max;
}

void build_bvh(bvh_node* nodes, int idx, sphere* l, int n) {
    assert(n >= lane_size_spheres);
    nodes[idx] = bvh_node(minof(l, n), maxof(l, n));

    if (n > lane_size_spheres) {
        const unsigned int axis = nodes[idx].split_axis();
        if (axis == 0)
            qsort(l, n, sizeof(sphere), box_x_compare);
        else if (axis == 1)
            qsort(l, n, sizeof(sphere), box_y_compare);
        else
            qsort(l, n, sizeof(sphere), box_z_compare);

        build_bvh(nodes, idx * 2, l, n / 2);
        build_bvh(nodes, idx * 2 + 1, l + n / 2, n / 2);
    }
}

bvh_node* build_bvh(sphere* l, int n, int& bvh_size) {
    bvh_size = 2 * n / lane_size_spheres;
    bvh_node* nodes = new bvh_node[bvh_size];
    build_bvh(nodes, 1, l, n);

    return nodes;
}

__device__ float hit_bbox(const bvh_node& node, const ray& r, float t_max) {
    float t_min = 0.001f;
    for (int a = 0; a < 3; a++) {
        float invD = 1.0f / r.direction()[a];
        float t0 = (node.min()[a] - 1 - r.origin()[a]) * invD;
        float t1 = (node.max()[a] + 1 - r.origin()[a]) * invD;
        if (invD < 0.0f) {
            float tmp = t0; t0 = t1; t1 = tmp;
        }
        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;
        if (t_max <= t_min)
            return FLT_MAX;
    }

    return t_min;
}
