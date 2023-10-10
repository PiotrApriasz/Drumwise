using System.Security.AccessControl;
using Drumwise.Domain.Common;

namespace Drumwise.Domain.Events.Homework;

public class HomeworkFinishedEvent(Entities.Homework item) : BaseEvent
{
    public Entities.Homework Item { get; set; } = item;
}