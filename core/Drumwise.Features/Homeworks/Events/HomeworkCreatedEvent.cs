using Drumwise.Application.Common.Models;
using Drumwise.Application.Entities;

namespace Drumwise.Features.Homeworks.Events;

public class HomeworkCreatedEvent(Homework item, string clientUrl) : BaseEvent
{
    public Homework Item { get; set; } = item;
    public string ClientUrl { get; set; } = clientUrl;
}