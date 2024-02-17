using Drumwise.Application.Common.Interfaces;
using Drumwise.Application.Common.Models.Settings;
using MailKit.Net.Smtp;
using Microsoft.AspNetCore.Identity;
using MimeKit;

namespace Drumwise.Infrastructure.Identity;

public class IdentityEmailSender(IMailSender mailSender, IFileService fileService) : IEmailSender<ApplicationUser>
{
    public async Task SendConfirmationLinkAsync(ApplicationUser user, string email, string confirmationLink)
    {
        var emailConfirmationTemplate = await fileService
            .GetTemplateAsync("EmailTemplates", "EmailConfirmationTemplate", "html")
            .ConfigureAwait(false);

        emailConfirmationTemplate = emailConfirmationTemplate
            .Replace("[EMAIL_ADDRESS]", email)
            .Replace("[CONFIRMATION_LINK]", confirmationLink);

        await mailSender.SendMailAsync(email, "Confirm your email for Drumwise", 
            emailConfirmationTemplate, "html").ConfigureAwait(false);
    }

    public Task SendPasswordResetLinkAsync(ApplicationUser user, string email, string resetLink)
    {
        throw new NotImplementedException();
    }

    public Task SendPasswordResetCodeAsync(ApplicationUser user, string email, string resetCode)
    {
        throw new NotImplementedException();
    }
}