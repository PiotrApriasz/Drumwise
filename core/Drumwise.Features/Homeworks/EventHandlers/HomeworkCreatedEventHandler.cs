using System.Globalization;
using Drumwise.Application.Common.Constants;
using Drumwise.Application.Common.Interfaces;
using Drumwise.Features.Homeworks.Events;
using MediatR;
using NLog;

namespace Drumwise.Features.Homeworks.EventHandlers;

public class HomeworkCreatedEventHandler(IMailSender mailSender, IIdentityService identityService) 
    : INotificationHandler<HomeworkCreatedEvent>
{
    private static readonly ILogger Logger = LogManager.GetCurrentClassLogger();
    
    public async Task Handle(HomeworkCreatedEvent notification, CancellationToken cancellationToken)
    {
        Logger.Info("Homework Created Event: {DomainEvent}", notification.GetType().Name);
        
        var studentMail = await identityService.GetUserNameAsync(notification.Item.AssignedTo);
        var teacherMail = await identityService.GetUserNameAsync(notification.Item.CreatedBy);

        var studentFullName = await identityService.GetUserFullNameIfAvailable(notification.Item.AssignedTo);
        var teacherFullName = await identityService.GetUserFullNameIfAvailable(notification.Item.CreatedBy);

        studentFullName ??= studentMail;
        teacherFullName ??= teacherMail;
        
        var exerciseUrl = $"{notification.ClientUrl}/{CommonClientRoutes.Exercise}/{notification.Item.Id}";

        var emailData = new HomeworkCreatedMailData(studentFullName,
            teacherFullName,
            notification.Item.HomeworkTitle,
            notification.Item.Deadline.ToString("D"),
            exerciseUrl);

        await mailSender
            .SendMailAsync(studentMail!, "New Drumwise Exercise", emailData)
            .ConfigureAwait(false);
    }
}

public record HomeworkCreatedMailData(string? StudentId, 
    string? TeacherId, 
    string Title, 
    string Deadline, 
    string ExerciseLink) : IMailData;