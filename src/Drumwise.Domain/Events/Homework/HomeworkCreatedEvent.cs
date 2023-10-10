using Drumwise.Domain.Common;

namespace Drumwise.Domain.Events.Homework;

public class HomeworkCreatedEvent(Entities.Homework item) : BaseEvent
{
    public Entities.Homework Item { get; set; } = item;
}