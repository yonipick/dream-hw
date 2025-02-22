from datetime import datetime, timezone

from pydantic import BaseModel


class SharedBaseModel(BaseModel):  # type: ignore
    class Config:  # pylint: disable=too-few-public-methods
        # Don't force use of aliases in class instantiation/attribute access
        allow_population_by_field_name = True

        # Always serialize datetime in the ISO 8601 format, to preserve timezones
        json_encoders = {
            datetime: lambda date_time: date_time.replace(tzinfo=timezone.utc).isoformat()
            if not date_time.tzinfo
            else date_time.isoformat(),
        }
