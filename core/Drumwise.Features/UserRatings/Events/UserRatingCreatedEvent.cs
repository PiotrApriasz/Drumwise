using Drumwise.Application.Common.Models;
using Drumwise.Application.Entities;

namespace Drumwise.Features.UserRatings.Events;

public class UserRatingCreatedEvent(UserRating item) : BaseEvent
{
    public UserRating Item { get; set; } = item;
}