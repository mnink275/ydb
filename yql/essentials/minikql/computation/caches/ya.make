PROGRAM()

PEERDIR(
    ydb/library/actors/core

    contrib/restricted/boost/intrusive
    yql/essentials/public/udf/service/exception_policy
    yql/essentials/parser/pg_wrapper

    yql/essentials/minikql/computation
    yql/essentials/minikql/comp_nodes
)

CFLAGS(
    -mavx2
)

SRCS(
    main_actors.cpp
    # main.cpp
)

YQL_LAST_ABI_VERSION()

END()
