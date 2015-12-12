/*  gcc -O1 -o dijkstra_serial dijkstra_serial.c -lrt -lm -fopenmp  */

//General includes
#include <stdio.h>      //I/O
#include <stdlib.h>
#include <time.h>       //for code timing purposes
#include <math.h>

//Parallel includes
#include <omp.h>        //need OpenMP header file for parallelization

//CUDA includes

//Parameters; modify as needed
#define VERTICES 16384            //number of vertices
#define DENSITY 32              //minimum number of edges per vertex. DO NOT SET TO >= VERTICES/2
#define MAX_WEIGHT 100000         //max edge length + 1
#define INF_DIST 1000000000     //"infinity" initial value of each node
#define IMPLEMENTATIONS 2       //number of Dijkstra implementations
#define THREADS 4               //number of OMP threads
#define RAND_SEED 1234          //random seed
typedef float data_t;           //data type

#define CPG 3.611
#define GIG 1000000000

void main() {

    srand(RAND_SEED);

    //functions
    void setIntArrayValue(int* in_array, int array_size, int value);
    void setDataArrayValue(data_t* in_array, int array_size, data_t init_value);
    void initializeGraphZero(data_t* graph, int num_vertices);
    void constructGraphEdge(data_t* graph, int* edge_count, int num_vertices);

    //Dijkstra's implementations
    void dijkstraCPUSerial(data_t* graph, data_t* node_dist, int* parent_node, int* visited_node, int num_vertices, int v_start);               //serial Dijkstra
    void dijkstraCPUParallel(data_t* graph, data_t* node_dist, int* parent_node, int* visited_node, int num_vertices, int v_start);             //OpenMP Dijkstra
   
    //timing
    struct timespec diff(struct timespec start, struct timespec end);
    struct timespec start, end;                     //timespec
    struct timespec time_stamp[IMPLEMENTATIONS];

    /*************SETUP GRAPH*************/

    //declare variables and allocate memory
    int graph_size      = VERTICES*VERTICES*sizeof(data_t);     //memory in B required by adjacency matrix representation of graph
    int int_array       = VERTICES*sizeof(int);                 //memory in B required by array of vertex IDs. Vertices have int IDs.
    int data_array      = VERTICES*sizeof(data_t);              //memory in B required by array of vertex distances (depends on type of data used)
    data_t* graph       = (data_t*)malloc(graph_size);                  //graph itself
    data_t* node_dist   = (data_t*)malloc(data_array);                  //distances from source indexed by node ID
    int* parent_node    = (int*)malloc(int_array);                      //previous nodes on SP indexed by node ID
    int* edge_count     = (int*)malloc(int_array);                      //number of edges per node indexed by node ID
    int* visited_node   = (int*)malloc(int_array);                      //pseudo-bool if node has been visited indexed by node ID
    int *pn_matrix      = (int*)malloc(IMPLEMENTATIONS*int_array);      //matrix of parent_node arrays (one per each implementation)
    data_t* dist_matrix = (data_t*)malloc(IMPLEMENTATIONS*data_array);

    //initialize arrays and graph
    setIntArrayValue(edge_count, VERTICES, 0);          //no edges visited yet
    initializeGraphZero(graph, VERTICES);               //initialize edges to zero
    constructGraphEdge(graph, edge_count, VERTICES);    //create weighted edges and connected graph
    free(edge_count);                   //no longer needed

    /************RUN DIJKSTRA'S************/
    
    int i;                                          //iterator
    int origin = (rand() % VERTICES);               //starting vertex
    printf("Origin vertex: %d\n\n", origin);

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

    /*  PARALLEL (OPENMP) DIJKSTRA  */
    version++;
    printf("Running OpenMP...");
    clock_gettime(CLOCK_REALTIME, &start);
    dijkstraCPUParallel(graph, node_dist, parent_node, visited_node, VERTICES, origin);
    clock_gettime(CLOCK_REALTIME, &end);
    time_stamp[version] = diff(start, end);               //record time
    for (i = 0; i < VERTICES; i++) {                //record resulting parent array
        pn_matrix[version*VERTICES + i] = parent_node[i];
        dist_matrix[version*VERTICES + i] = node_dist[i];
    }
    printf("Done!\n");

    printf("\nVertices: %d", VERTICES);
    printf("\nDensity: %d", DENSITY);
    printf("\nMax Weight: %d", MAX_WEIGHT);
    printf("\n\nTime (cycles):\nSerial,OpenMP\n");
    for (i = 0; i < IMPLEMENTATIONS; i++) {
        printf("%ld,", (long int)( (double)(CPG)*(double)
            (GIG * time_stamp[i].tv_sec + time_stamp[i].tv_nsec)));
    }

    /*  ERROR CHECKING  */
    /***************ERROR CHECKING***************/
    printf("\n\nError checking:\n");

    printf("----Serial vs OPenMP:\n");
    int p_errors = 0, d_errors = 0;
    for (i = 0; i < VERTICES; i++) {
        if (pn_matrix[i] != pn_matrix[VERTICES + i]) {
            p_errors++;
        }
        if (dist_matrix[i] != dist_matrix[VERTICES + i]) {
            d_errors++;
            //printf("Error: Serial has %d, OMP has %d\n", dist_matrix[i], dist_matrix[VERTICES + i]);
        }
    }
    printf("--------%d parent errors found.\n", p_errors);
    printf("--------%d dist errors found.\n", d_errors);

}

/* get closest node to current node that hasn't been visited*/
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

/* OpenMP implementation of closestNode() */
int closestNodeOMP(data_t* node_dist, int* visited_node, int num_vertices) {
    data_t min_dist = INF_DIST + 1;    //initialize global distance to INF_DIST + 1
    int min_node = -1;              //not an actual node, but will get overwritten
    int min_dist_thread, min_node_thread;

    int vertex;
    /*  Each thread works on a subset of elements in the node distance array and finds a min in the subset.
    At the end, each thread compares their min to the global min and the min of mins is selected.   */
    omp_set_num_threads(THREADS);
    #pragma omp parallel private(min_dist_thread, min_node_thread) shared(node_dist, visited_node)
    {
        min_dist_thread = min_dist;             //thread-local minimum distance
        min_node_thread = min_node;             //thread-local node ID
        #pragma omp barrier                     //ensure all threads load data

        #pragma omp for nowait                      //each thread finds its own minimum node and distance
        for (vertex = 0; vertex < num_vertices; vertex++) {            //at the end, pick min of mins
            if ((node_dist[vertex] < min_dist_thread) && (visited_node[vertex] == 0)) {
                min_dist_thread = node_dist[vertex];
                min_node_thread = vertex;
            }
        }
        #pragma omp critical                    //threads compare/update value one by one (CRITICAL SECTION)
        {
            if (min_dist_thread < min_dist) {
                min_dist = min_dist_thread;
                min_node = min_node_thread;
            }
        }
    }
    return min_node;
}

/*  Construct graph with randomized edges.  */
void constructGraphEdge(data_t* graph, int* edge_count, int num_vertices) {
    int i;                  //iterator
    int rand_vertex;        //random previous vertex
    int curr_num_edges;     //current number of edges per vertex
    int num_edges;          //edges per vertex
    data_t edge, weight;    //edge chance and weight

    //initialize a connected graph
    for (i = 1; i < num_vertices; i++) {
        rand_vertex = (rand() % i);                     //select a random previous vertex to create a connected graph
        weight = (rand() % MAX_WEIGHT) + 1;             //random (non-zero) weight
        graph[rand_vertex*num_vertices + i] = weight;   //set edge weights
        graph[i*num_vertices + rand_vertex] = weight;
        edge_count[i] += 1;                             //increment edge counts for each vertex
        edge_count[rand_vertex] += 1;
    }

    //add additional edges until DENSITY reached for all vertices
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
}

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

/*  Print int array elements    */
void checkArray(int* a, int length) {
    int i;
    printf("Proof: ");
    for (i = 0; i < length; i++) {
        printf("%d, ", a[i]);
    }
    printf("\n");
}

/*   Difference in two timespec objects   */
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
    void checkArray(int* a, int length);

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

/*  Parallel implementation of Dijkstra's algorithm. Makes use of OpenMP to parallelize
Dijkstra's algorithm, so should have better performance than serial code.   */
void dijkstraCPUParallel(data_t* graph, data_t* node_dist, int* parent_node, int* visited_node, int num_vertices, int v_start) {
    //functions
    void setIntArrayValue(int* in_array, int array_size, int init_value);
    void setDataArrayValue(data_t* in_array, int array_size, data_t init_value);
    int closestNodeOMP(data_t* node_dist, int* visited_node, int num_vertices);        //utilize OpenMP reduction to find closest
    int closestNode(data_t* node_dist, int* visited_node, int num_vertices);

    //reset/clear data from previous runs
    setDataArrayValue(node_dist, VERTICES, INF_DIST);     //all node distances are infinity
    setIntArrayValue(parent_node, VERTICES, -1);          //parent nodes are -1 (no parents yet)
    setIntArrayValue(visited_node, VERTICES, 0);          //no nodes have been visited

    node_dist[v_start] = 0;                     //start distance is zero; ensures it will be first pulled out

    int i, next;
    for (i = 0; i < num_vertices; i++) {
        int curr_node = closestNodeOMP(node_dist, visited_node, num_vertices);      //closest node
        visited_node[curr_node] = 1;

        /*  Split work of updating neighbor vertices distances across multiple threads.
        Since each vertex update depends only on the previously computed current node,
        there is no need to synchronize.    */
        omp_set_num_threads(THREADS);
        int new_dist;
        #pragma omp parallel shared(graph,node_dist) 
        {
            #pragma omp for private(new_dist,next)
            for (next = 0; next < num_vertices; next++) {
                new_dist = node_dist[curr_node] + graph[curr_node*num_vertices + next];
                if ((visited_node[next] != 1)                                   //if not visited                NO CONFLICTS (READ ONLY COMPARE)
                    && (graph[curr_node*num_vertices + next] != (data_t)(0))    //and edge weight != 0          NO CONFLICTS (READ ONLY COMPARE
                    && (new_dist < node_dist[next])) {                          //and new_dist < old dist       NO CONFLICTS (EACH THREAD GETS DIFFERENT next AND new_dist)
                    node_dist[next] = new_dist;        //update distance
                    parent_node[next] = curr_node;     //update predecessor
                }
            }
            #pragma omp barrier
        }
    }
}