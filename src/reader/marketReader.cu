#include "marketReader.cuh"

void readFile(char *fileLoc, GraphProperties *h_graph_prop, thrust::host_vector<unsigned long> &h_source, thrust::host_vector<unsigned long> &h_destination, thrust::host_vector<unsigned long> &h_source_degrees, unsigned long type) {

    FILE *ptr;
    char buff[600];
    ptr = fopen(fileLoc, "a+");

    if (NULL == ptr)
    {
        printf("file can't be opened \n");
    }

    int dataFlag = 0;
    unsigned long index = 0;

    while (fgets(buff, 400, ptr) != NULL)
    {

        // ignore commented lines
        if (buff[0] == '%')
            continue;
        // reading edge values
        else if (dataFlag)
        {

            // printf("%s", buff);
            unsigned long source;
            unsigned long destination;
            int i = 0, j = 0;
            char temp[100];

            while (buff[i] != ' ')
                temp[j++] = buff[i++];
            source = strtol(temp, NULL, 10);
            // printf("%lu ", nodeID);
            memset(temp, '\0', 100);

            i++;
            j = 0;
            // while( buff[i] != ' ' )
            while ((buff[i] != '\0') && (buff[i] != ' '))
                temp[j++] = buff[i++];
            destination = strtol(temp, NULL, 10);
            // printf("%.8Lf ", x);
            memset(temp, '\0', 100);
            // h_edges[i] = ;
            if ((index % 500000000) == 0)
                printf("%lu and %lu\n", source, destination);
            // if((source >= h_graph_prop->xDim) || (destination >= h_graph_prop->xDim))
            //     printf("%lu and %lu\n", source, destination);

            h_source[index] = source;
            h_destination[index++] = destination;
            h_source_degrees[source - 1]++;

            // below part makes it an undirected graph
            if (!DIRECTED)
            {

                h_source[index] = destination;
                h_destination[index++] = source;
                h_source_degrees[destination - 1]++;
            }
        }
        // reading xDim, yDim, and total_edges
        else
        {

            unsigned long xDim, yDim, total_edges;

            int i = 0, j = 0;
            char temp[100];

            while (buff[i] != ' ')
                temp[j++] = buff[i++];
            xDim = strtol(temp, NULL, 10);
            // printf("%lu ", nodeID);
            memset(temp, '\0', 100);

            i++;
            j = 0;
            while (buff[i] != ' ')
                temp[j++] = buff[i++];
            yDim = strtol(temp, NULL, 10);
            // printf("%.8Lf ", x);
            memset(temp, '\0', 100);

            i++;
            j = 0;
            while (buff[i] != '\0')
                temp[j++] = buff[i++];
            total_edges = strtol(temp, NULL, 10);
            // printf("%.8Lf\n", y);
            memset(temp, '\0', 100);

            if (DIRECTED)
                printf("xDim = %lu, yDim = %lu, Total Edges = %lu\n", xDim, yDim, total_edges);
            else
                printf("xDim = %lu, yDim = %lu, Total Edges = %lu\n", xDim, yDim, total_edges * 2);

            h_graph_prop->xDim = xDim;
            h_graph_prop->yDim = yDim;
            // total edges doubles since undirected graph
            if (DIRECTED)
                h_graph_prop->total_edges = total_edges;
            else
                h_graph_prop->total_edges = total_edges * 2;

            h_source.resize(h_graph_prop->total_edges);
            h_destination.resize(h_graph_prop->total_edges);
           
            h_source_degrees.resize(h_graph_prop->xDim);

            dataFlag = 1;
        }
    }

    fclose(ptr);

}