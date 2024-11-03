PROGRAM()

PEERDIR(
    ydb/library/yql/minikql
    ydb/library/yql/parser/pg_wrapper
    ydb/library/yql/public/udf/service/exception_policy
    contrib/libs/tbb
    ydb/core/ymq/actor
)

SRCS(
    main.cpp
)

YQL_LAST_ABI_VERSION()

END()
