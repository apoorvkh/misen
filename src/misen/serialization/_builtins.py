def register_serializer_for_builtin_types() -> None:
    import datetime
    import decimal
    import enum
    import sys
    import uuid

    from . import msgspec_serializer, register_serializer, serialize

    register_serializer(
        [
            type(None),
            bool,
            int,
            float,
            str,
            bytes,
            bytearray,
            datetime.datetime,
            datetime.date,
            datetime.time,
            datetime.timedelta,
            uuid.UUID,
            decimal.Decimal,
            enum.Enum,
            enum.IntEnum,
            enum.Flag,
            enum.IntFlag,
        ],
        msgspec_serializer,
    )

    if sys.version_info >= (3, 11):
        register_serializer([enum.StrEnum], msgspec_serializer)

    register_serializer(
        [list, tuple, set, frozenset],
        lambda o: msgspec_serializer([serialize(i) for i in o]),
    )

    register_serializer(
        [dict],
        lambda o: msgspec_serializer(
            {serialize(k).decode(encoding="latin-1"): serialize(v) for k, v in o.items()}
        ),
    )
