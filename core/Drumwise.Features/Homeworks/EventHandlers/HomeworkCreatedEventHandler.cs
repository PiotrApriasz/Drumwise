using Drumwise.Features.Homeworks.Events;
using MediatR;
using NLog;
using NLog.Fluent;

namespace Drumwise.Features.Homeworks.EventHandlers;

public class HomeworkCreatedEventHandler : INotificationHandler<HomeworkCreatedEvent>
{
    private readonly ILogger _logger = LogManager.GetCurrentClassLogger();
    
    public Task Handle(HomeworkCreatedEvent notification, CancellationToken cancellationToken)
    {
        // TODO: Send notification (eg. email) to user to which homework has been assigned
        
        _logger.Info("Drumwise Domain Event: {DomainEvent}", notification.GetType().Name);
        
        return Task.CompletedTask;
    }
}