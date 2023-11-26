using Drumwise.Application.Common.Models;
using Drumwise.Application.Entities;

namespace Drumwise.Features.Homeworks.Events;

public class HomeworkCreatedEvent(Homework item) : BaseEvent
{
    public Homework Item { get; set; } = item;
}