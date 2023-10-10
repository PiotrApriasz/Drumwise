using Drumwise.Domain.Common;

namespace Drumwise.Domain.Events.UserRating;

public class UserRatingCreatedEvent(Entities.UserRating item) : BaseEvent
{
    public Entities.UserRating Item { get; set; } = item;
}