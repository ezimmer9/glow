add_library(ExecutionEngine
              ExecutionEngine.cpp)

target_link_libraries(ExecutionEngine
                      PRIVATE
                        Backend
                        Backends
                        Optimizer
                        Base
                        Graph)
