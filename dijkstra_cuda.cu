/*  nvcc -O1 -o dijkstra_cuda dijkstra_cuda.cu -Xcompiler -lrt -lm  */

//General includes
#include <stdio.h>      //I/O
#include <stdlib.h>
#include <time.h>       //for code timing purposes
#include <math.h>


//Parameters; modify as needed
#define VERTICES 16384           //number of vertices
#define DENSITY 16              //minimum number of edges per vertex. DO NOT SET TO >= VERTICES
#define MAX_WEIGHT 1000000      //max edge length + 1
#define INF_DIST 1000000000     //"infinity" initial value of each node
#define CPU_IMP 1               //number of Dijkstra implementations (non-GPU)
#define GPU_IMP 1               //number of Dijkstra implementations (GPU)
#define THREADS 2               //number of OMP threads
#define RAND_SEED 1234          //random seed
#define THREADS_BLOCK 512

typedef float data_t;             //data type

//CPU parameters for serial implementation
#define CPG 2.53
#define GIG 1000000000

//Check for CUDA errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

int main() {

    srand(RAND_SEED);   //random seed

    //functions
    void setIntArrayValue(int* in_array, int array_size, int value);                //initialize 1D int array
    void setDataArrayValue(data_t* in_array, int array_size, data_t init_value);    //initialize 1D data_t array
    void initializeGraphZero(data_t* graph, int num_vertices);                      //initialize VxV graph to zero
    void constructGraphEdge(data_t* graph, int* edge_count, int num_vertices);      //construct VxV graph
    void checkArray(int* a, int length);                                            //print 1D int array
    void checkArrayData(data_t* a, int length);

    //Dijkstra's implementations
    void dijkstraCPUSerial(data_t* graph, data_t* node_dist, int* parent_node, int* visited_node, int num_vertices, int v_start);   //serial Dijkstra
    __global__ void closestNodeCUDA(data_t* node_dist, int* visited_node, int* global_closest, int num_vertices);                   //Dijkstra CUDA Pt. 1
    __global__ void cudaRelax(data_t* graph, data_t* node_dist, int* parent_node, int* visited_node, int* source);                  //Dijkstra CUDA Pt. 2

    //timing
    struct timespec diff(struct timespec start, struct timespec end);
    struct timespec start, end;                     //timespec
    struct timespec time_stamp[CPU_IMP];

    /*************SETUP GRAPH*************/

    //declare variables and allocate memory
    int graph_size      = VERTICES*VERTICES*sizeof(data_t);             //memory in B required by adjacency matrix representation of graph
    int int_array       = VERTICES*sizeof(int);                         //memory in B required by array of vertex IDs. Vertices have int IDs.
    int data_array      = VERTICES*sizeof(data_t);                      //memory in B required by array of vertex distances (depends on type of data used)
    data_t* graph       = (data_t*)malloc(graph_size);                  //graph itself
    data_t* node_dist   = (data_t*)malloc(data_array);                  //distances from source indexed by node ID
    int* parent_node    = (int*)malloc(int_array);                      //previous nodes on SP indexed by node ID
    int* edge_count     = (int*)malloc(int_array);                      //number of edges per node indexed by node ID
    int* visited_node   = (int*)malloc(int_array);                      //pseudo-bool if node has been visited indexed by node ID
    int *pn_matrix      = (int*)malloc((CPU_IMP+GPU_IMP)*int_array);    //matrix of parent_node arrays (one per each implementation)
    data_t* dist_matrix = (data_t*)malloc((CPU_IMP + GPU_IMP)*data_array);

    printf("Variables created, allocated\n");

    //CUDA mallocs
    data_t* gpu_graph;
    data_t* gpu_node_dist;
    int* gpu_parent_node;
    int* gpu_visited_node;
    CUDA_SAFE_CALL(cudaMalloc((void**)&gpu_graph, graph_size));
    CUDA_SAFE_CALL(cudaMalloc((void**)&gpu_node_dist, data_array));
    CUDA_SAFE_CALL(cudaMalloc((void**)&gpu_parent_node, int_array));
    CUDA_SAFE_CALL(cudaMalloc((void**)&gpu_visited_node, int_array));

    //for closest vertex
    int* closest_vertex = (int*)malloc(sizeof(int));
    int* gpu_closest_vertex;
    closest_vertex[0] = -1;
    CUDA_SAFE_CALL(cudaMalloc((void**)&gpu_closest_vertex, (sizeof(int))));
    CUDA_SAFE_CALL(cudaMemcpy(gpu_closest_vertex, closest_vertex, sizeof(int), cudaMemcpyHostToDevice));

    //initialize arrays
    //node_dist, parent_node and visited_node done within voidDijkstraCPUSerial
    //same graph is used for ALL versions of dijkstra's (serial, parallel, CUDA)

    setIntArrayValue(edge_count, VERTICES, 0);          //no edges visited yet
    setDataArrayValue(node_dist, VERTICES, INF_DIST);   //all node distances are infinity
    setIntArrayValue(parent_node, VERTICES, -1);        //parent nodes are -1 (no parents yet)
    setIntArrayValue(visited_node, VERTICES, 0);        //no nodes have been visited
    initializeGraphZero(graph, VERTICES);               //initialize edges to zero
    constructGraphEdge(graph, edge_count, VERTICES);    //create weighted edges and connected graph
    free(edge_count);                   //no longer needed
    printf("Variables initialized.\n");

    /************RUN DIJKSTRA'S************/

    int i;                                          //iterator
    int origin = (rand() % VERTICES);               //starting vertex
    printf("Origin vertex: %d\n", origin);

    /*  SERIAL DIJKSTRA  */
    int version = 0;
    printf("Running serial...");
    clock_gettime(CLOCK_REALTIME, &start);
    dijkstraCPUSerial(graph, node_dist, parent_node, visited_node, VERTICES, origin);
    clock_gettime(CLOCK_REALTIME, &end);
    time_stamp[version] = diff(start, end);               //record time
    for (i = 0; i < VERTICES; i++) {                //record resulting parent array
        pn_matrix[version*VERTICES + i] = parent_node[i];
        dist_matrix[version*VERTICES + i] = node_dist[i];
    }
    printf("Done!\n");


    /*  CUDA DIJKSTRA  */
    version++;
    cudaEvent_t exec_start, exec_stop;              //timer for execution only
    float elapsed_exec;                             //elapsed time
    CUDA_SAFE_CALL(cudaEventCreate(&exec_start));
    CUDA_SAFE_CALL(cudaEventCreate(&exec_stop));

    //need to reset data from previous run, since serial and parallel versions do this automatically
    setDataArrayValue(node_dist, VERTICES, INF_DIST);       //all node distances are infinity
    setIntArrayValue(parent_node, VERTICES, -1);            //parent nodes are -1 (no parents yet)
    setIntArrayValue(visited_node, VERTICES, 0);            //no nodes have been visited
    node_dist[origin] = 0;                                  //start distance is zero; ensures it will be first pulled out


    //gpu source        cpu source      memory size     HtD or DtH
    CUDA_SAFE_CALL(cudaMemcpy(gpu_graph, graph, graph_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(gpu_node_dist, node_dist, data_array, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(gpu_parent_node, parent_node, int_array, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(gpu_visited_node, visited_node, int_array, cudaMemcpyHostToDevice));

    //Min:  One thread checks for closest vertex. Ideally there would be multiple threads working in
    //  parallel, but due to compiler issues with prallelized-reduction functions this is being used as a backup.
    dim3 gridMin(1, 1, 1);
    dim3 blockMin(1, 1, 1);

    //Relax: Each thread is responsible for relaxing from a shared, given vertex
    //  to one other vertex determined by the ID of the thread. Since each thread handles
    //  a different vertex, there's no RaW or WaR data hazards; all that's needed is a
    //  __syncthreads(); call at the end to ensure all either update or do nothing.
    dim3 gridRelax(VERTICES / THREADS_BLOCK, 1, 1);
    dim3 blockRelax(THREADS_BLOCK, 1, 1);           
    

    //Show Nvidia GPU info (code from http://devblogs.nvidia.com/parallelforall/how-query-device-properties-and-handle-errors-cuda-cc/)
    /*int nDevices;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&nDevices));
    for (i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, i));
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
            2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
    }*/

    CUDA_SAFE_CALL(cudaEventRecord(exec_start));
    for (int i = 0; i < VERTICES; i++) {
        closestNodeCUDA <<<gridMin, blockMin>>>(gpu_node_dist, gpu_visited_node, gpu_closest_vertex, VERTICES);                 //find min
        cudaRelax <<<gridRelax, blockRelax>>>(gpu_graph, gpu_node_dist, gpu_parent_node, gpu_visited_node, gpu_closest_vertex); //relax
    }
    CUDA_SAFE_CALL(cudaEventRecord(exec_stop));
    
    //save data in PN, ND matrices
    CUDA_SAFE_CALL(cudaMemcpy(node_dist, gpu_node_dist, data_array, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(parent_node, gpu_parent_node, int_array, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(visited_node, gpu_visited_node, int_array, cudaMemcpyDeviceToHost));
    for (i = 0; i < VERTICES; i++) {                //record resulting parent array and node distance
        pn_matrix[version*VERTICES + i] = parent_node[i];
        dist_matrix[version*VERTICES + i] = node_dist[i];
    }

    //free memory
    CUDA_SAFE_CALL(cudaFree(gpu_graph));
    CUDA_SAFE_CALL(cudaFree(gpu_node_dist));
    CUDA_SAFE_CALL(cudaFree(gpu_parent_node));
    CUDA_SAFE_CALL(cudaFree(gpu_visited_node));


    printf("\nVertices: %d", VERTICES);
    printf("\nDensity: %d", DENSITY);
    printf("\nMax Weight: %d", MAX_WEIGHT);
    printf("\n\nSerial cycles: \n");
    for (i = 0; i < CPU_IMP; i++) {
        printf("%ld", (long int)((double)(CPG)*(double)
            (GIG * time_stamp[i].tv_sec + time_stamp[i].tv_nsec)));
    }

    //calculate elapsed time
    CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_exec, exec_start, exec_stop));        //elapsed execution time
    printf("\n\nCUDA Time (ms): %7.9f\n", elapsed_exec);

    /***************ERROR CHECKING***************/
    printf("\n\nError checking:\n");

    printf("----Serial vs CUDA:\n");
    int p_errors = 0, d_errors = 0;
    /*for (i = 0; i < VERTICES; i++) {
        if (pn_matrix[i] != pn_matrix[VERTICES + i]) {
            p_errors++;
        }
        if (dist_matrix[i] != dist_matrix[VERTICES + i]) {
            d_errors++;
            //printf("Error: Serial has %d, OMP has %d\n", dist_matrix[i], dist_matrix[VERTICES + i]);
        }
    }*/
    printf("--------%d parent errors found.\n", p_errors);
    printf("--------%d dist errors found.\n", d_errors);
}

/********FUNCTIONS*********/

/*  Initialize elements of a 1D int array with an initial value   */
void setIntArrayValue(int* in_array, int array_size, int init_value) {
    int i;
    for (i = 0; i < array_size; i++) {
        in_array[i] = init_value;
    }
}

/*  Initialize elements of a 1D data_t array with an initial value   */
void setDataArrayValue(data_t* in_array, int array_size, data_t init_value) {
    int i;
    for (i = 0; i < array_size; i++) {
        in_array[i] = init_value;
    }
}

/*  Construct graph with no edges or weights     */
void initializeGraphZero(data_t* graph, int num_vertices) {
    int i, j;

    for (i = 0; i < num_vertices; i++) {
        for (j = 0; j < num_vertices; j++) {           //weight of all edges initialized to 0
            graph[i*num_vertices + j] = (data_t)0;
        }
    }
}

/*  Construct graph with randomized edges.  */
void constructGraphEdge(data_t* graph, int* edge_count, int num_vertices) {
    /*  Guarantees:
    -A fully connected, undirected graph.
    -Non-negative edge weights
    -A minimum degree of DEGREE for each vertex
    */

    int closestNode(data_t* node_dist, int* visited_node, int num_vertices);
    int i;                  //iterator
    int rand_vertex;        //random previous vertex
    int curr_num_edges;     //current number of edges per vertex
    data_t weight;    //edge chance and weight

    //initialize a connected graph
    printf("Initializing a connected graph...");
    for (i = 1; i < num_vertices; i++) {
        rand_vertex = (rand() % i);                     //select a random previous vertex to create a connected graph
        weight = (rand() % MAX_WEIGHT) + 1;             //random (non-zero) weight
        graph[rand_vertex*num_vertices + i] = weight;   //set edge weights
        graph[i*num_vertices + rand_vertex] = weight;
        edge_count[i] += 1;                             //increment edge counts for each vertex
        edge_count[rand_vertex] += 1;
    }
    printf("done!\n");

    //add additional edges until DENSITY reached for all vertices
    printf("Checking density...");
    for (i = 0; i < num_vertices; i++) {    //for each vertex
        curr_num_edges = edge_count[i];         //current number of edges (degree) of vertex
        while (curr_num_edges < DENSITY) {      //add additional edges if number of edges < DENSITY
            rand_vertex = (rand() % num_vertices);  //choose any random vertex
            weight = (rand() % MAX_WEIGHT) + 1;     //choose a random (non-zero) weight
            if ((rand_vertex != i) && (graph[i*num_vertices + rand_vertex] == 0)) { //add edge if not trying to connect to itself and no edge currently exists
                graph[i*num_vertices + rand_vertex] = weight;
                graph[rand_vertex*num_vertices + i] = weight;
                edge_count[i] += 1;
                curr_num_edges++;               //one additional edge constructed
            }
        }
    }
    printf("done!\n");
}

/*  Get closest node to current node that hasn't been visited   */
int closestNode(data_t* node_dist, int* visited_node, int num_vertices) {
    data_t dist = INF_DIST + 1;    //set start to infinity+1, so guaranteed to pull out at least one node
    int node = -1;              //closest non-visited node
    int i;                      //iterator

    for (i = 0; i < num_vertices; i++) {
        if ((node_dist[i] < dist) && (visited_node[i] == 0)) {  //if closer and not visited
            node = i;               //select node
            dist = node_dist[i];    //new closest distance
        }
    }
    return node;    //return closest node
}

/*  Print int array elements    */
void checkArray(int* a, int length) {
    int i;
    printf("Proof: ");
    for (i = 0; i < length; i++) {
        printf("%d, ", a[i]);
    }
    printf("\n\n");
}

void checkArrayData(data_t* a, int length) {
    int i;
    printf("Proof: ");
    for (i = 0; i < length; i++) {
        printf("%f, ", a[i]);
    }
    printf("\n\n");
}

/*  Difference in two timespec objects   */
struct timespec diff(struct timespec start, struct timespec end)
{
    struct timespec temp;
    if ((end.tv_nsec - start.tv_nsec)<0) {
        temp.tv_sec = end.tv_sec - start.tv_sec - 1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    }
    else {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    return temp;
}

/****************DIJKSTRA'S ALGORITHM IMPLEMENTATIONS****************/
/*  Serial implementation of Dijkstra's algorithm. Not designed to be particularly efficient;
just nice to have a baseline comparison.    */
void dijkstraCPUSerial(data_t* graph, data_t* node_dist, int* parent_node, int* visited_node, int num_vertices, int v_start) {

    //functions
    void setIntArrayValue(int* in_array, int array_size, int init_value);
    void setDataArrayValue(data_t* in_array, int array_size, data_t init_value);
    int closestNode(data_t* node_dist, int* visited_node, int num_vertices);

    //reset/clear data from previous runs
    setDataArrayValue(node_dist, VERTICES, INF_DIST);     //all node distances are infinity
    setIntArrayValue(parent_node, VERTICES, -1);          //parent nodes are -1 (no parents yet)
    setIntArrayValue(visited_node, VERTICES, 0);          //no nodes have been visited
    node_dist[v_start] = 0;                     //start distance is zero; ensures it will be first pulled out

    int i, next;
    for (i = 0; i < num_vertices; i++) {
        int curr_node = closestNode(node_dist, visited_node, num_vertices); //get closest node not visited
        visited_node[curr_node] = 1;                                        //set node retrieved as visited
        /*
        Requirements to update neighbor's distance:
        -Neighboring node has not been visited.
        -Edge exists between current node and neighbor node
        -dist[curr_node] + edge_weight(curr_node, next_node) < dist[next_node]
        */
        for (next = 0; next < num_vertices; next++) {
            int new_dist = node_dist[curr_node] + graph[curr_node*num_vertices + next];
            if ((visited_node[next] != 1)
                && (graph[curr_node*num_vertices + next] != (data_t)(0))
                && (new_dist < node_dist[next])) {
                node_dist[next] = new_dist;        //update distance
                parent_node[next] = curr_node;     //update predecessor
            }
        }
    }
}

/*  CUDA implementation of Dijkstra's algorithm. Utilizes Nvidia GPUs to accelerate
computation. Done correctly, it should have noticeably improved performance versus
serial and even OpenMP.

    The code is split into two parts:
    1: closestNodaCUDA():   Uses a single thread in a single block to find the minimum.
    2: cudaRelax():         Uses multiple threads in multiple blocks to relax edges from a source to a destination
*/


__global__ void closestNodeCUDA(data_t* node_dist, int* visited_node, int* global_closest, int num_vertices) {
    data_t dist = INF_DIST + 1;
    int node = -1;
    int i;

    for (i = 0; i < num_vertices; i++) {
        if ((node_dist[i] < dist) && (visited_node[i] != 1)) {
            dist = node_dist[i];
            node = i;
        }
    }

    global_closest[0] = node;
    visited_node[node] = 1;
}

__global__ void cudaRelax(data_t* graph, data_t* node_dist, int* parent_node, int* visited_node, int* global_closest) {
    int next = blockIdx.x*blockDim.x + threadIdx.x;    //global ID
    int source = global_closest[0];

    data_t edge = graph[source*VERTICES + next];
    data_t new_dist = node_dist[source] + edge;

    if ((edge != 0) &&
        (visited_node[next] != 1) &&
        (new_dist < node_dist[next])) {
        node_dist[next] = new_dist;
        parent_node[next] = source;
    }

}