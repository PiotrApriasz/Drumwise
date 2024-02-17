using System.Globalization;
using Drumwise.Application.Common.Constants;
using Drumwise.Application.Common.Interfaces;
using Drumwise.Features.Homeworks.Events;
using MediatR;
using NLog;

namespace Drumwise.Features.Homeworks.EventHandlers;

public class HomeworkCreatedEventHandler(IMailSender mailSender, IIdentityService identityService, IFileService fileService) 
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
        
        var emailTemplate = await fileService
            .GetTemplateAsync("EmailTemplates", "HomeworkCreatedTemplate", "html")
            .ConfigureAwait(false);

        emailTemplate = emailTemplate
            .Replace("[STUDENT_ID]", studentFullName)
            .Replace("[TEACHER_ID]", teacherFullName)
            .Replace("[TITLE]", notification.Item.HomeworkTitle)
            .Replace("[DEADLINE]", notification.Item.Deadline.ToString("D"))
            .Replace("[EXERCISE_LINK]", exerciseUrl);

        await mailSender
            .SendMailAsync(studentMail!, "New Drumwise Exercise", emailTemplate, "html")
            .ConfigureAwait(false);
    }
}