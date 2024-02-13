using System.ComponentModel.DataAnnotations;
using System.Security.Claims;
using System.Text;
using System.Text.Encodings.Web;
using Drumwise.Application.Common.Errors;
using Drumwise.Application.Common.Exceptions;
using Drumwise.Application.Common.Interfaces;
using Drumwise.Application.Common.Models;
using Drumwise.Application.Common.Models.Identity;
using FluentValidation.Results;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.HttpResults;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Routing;
using Microsoft.AspNetCore.WebUtilities;
using Microsoft.EntityFrameworkCore;
using ValidationException = FluentValidation.ValidationException;

namespace Drumwise.Infrastructure.Identity;

public class IdentityService(UserManager<ApplicationUser> userManager,
        IUserClaimsPrincipalFactory<ApplicationUser> userClaimsPrincipalFactory,
        IAuthorizationService authorizationService,
        IUserStore<ApplicationUser> userStore,
        IEmailSender<ApplicationUser> emailSender,
        LinkGenerator linkGenerator)
    : IIdentityService
{
    private static readonly EmailAddressAttribute EmailAddressAttribute = new();
    
    public  async Task<string?> GetUserNameAsync(string userId)
    {
        var user = await userManager.Users.FirstAsync(u => u.Id == userId);
        return user.UserName;
    }

    // Clients should use that endpoint instead of register endpoint from ASP .NET
    // Identity API endpoint. I implemented that because clients have to call email
    // confirmation endpoint themselves (with original register endpoint generated
    // confirmation url points to web api without option to redirect to client).
    // With that way, client generates confirmation url which points to specific
    // client's page and when user hit that page, client send request to web api to
    // confirm an email.
    public async Task<Result> CustomUserRegister(UserRegisterDataRequest userRegisterDataRequest, 
        HttpContext context)
    {
        // TODO: Implement validation and sanitization of clientEmailConfirmationUrl
        
        var clientEmailConfirmationUrl = context.Request.Headers["X-Client-URL"].FirstOrDefault();
        
        var email = userRegisterDataRequest.Email;

        if (string.IsNullOrEmpty(email) || !EmailAddressAttribute.IsValid(email))
        {
            return Result.Failure(IdentityErrors.InvalidEmail(email), ResultType.BadRequest);
        }
        
        var emailStore = (IUserEmailStore<ApplicationUser>)userStore;

        var user = new ApplicationUser();
        await userStore.SetUserNameAsync(user, email, CancellationToken.None);
        await emailStore.SetEmailAsync(user, email, CancellationToken.None);
        var result = await userManager.CreateAsync(user, userRegisterDataRequest.Password);

        if (!result.Succeeded)
        {
            return Result.Failure(IdentityErrors.IdentityError(result.Errors), ResultType.BadRequest);
        }

        await SendConfirmationEmailAsync(user, email, clientEmailConfirmationUrl!);
        
        return Result.Success(ResultType.NoContent);
    }

    public async Task<Result> RegisterAdditionalUserData(AdditionalUserDataRequest additionalUserDataRequest, 
        ClaimsPrincipal claimsPrincipal)
    {
        if (await userManager.GetUserAsync(claimsPrincipal) is not { } user)
            return Result.Failure(IdentityErrors.UserNotFound, ResultType.NotFound);

        user.Name = additionalUserDataRequest.Name;
        user.Surname = additionalUserDataRequest.Surname;
        user.Experience = additionalUserDataRequest.Experience ?? 0;

        var result = await userManager.UpdateAsync(user);

        if (!result.Succeeded)
        {
            return Result.Failure(IdentityErrors.IdentityError(result.Errors), ResultType.BadRequest);
        }

        var roleResult = await userManager.AddToRoleAsync(user, additionalUserDataRequest.Role);

        if (!roleResult.Succeeded)
        {
            return Result.Failure(IdentityErrors.IdentityError(roleResult.Errors), ResultType.BadRequest);
        }

        return Result.Success(ResultType.NoContent);
    }

    public async Task<Result> DeleteUserAsync(string userId)
    {
        var user = userManager.Users.SingleOrDefault(u => u.Id == userId);
        return user != null ? await PerformDeleteUser(user) : Result.Success(ResultType.Ok);
    }

    private async Task<Result> PerformDeleteUser(ApplicationUser user)
    {
        var result = await userManager.DeleteAsync(user);
        return Result.Success(ResultType.Ok);
    }
    
    private async Task SendConfirmationEmailAsync(ApplicationUser user, string email, string url, bool isChange = false)
    {
        var code = isChange
            ? await userManager.GenerateChangeEmailTokenAsync(user, email)
            : await userManager.GenerateEmailConfirmationTokenAsync(user);
        code = WebEncoders.Base64UrlEncode(Encoding.UTF8.GetBytes(code));

        var userId = await userManager.GetUserIdAsync(user);
        var routeValues = new RouteValueDictionary()
        {
            ["userId"] = userId,
            ["code"] = code,
        };

        if (isChange)
        {
            // This is validated by the /confirmEmail endpoint on change.
            routeValues.Add("changedEmail", email);
        }

        var confirmEmailUrl = QueryHelpers.AddQueryString(url, 
            routeValues.ToDictionary(k => k.Key, k => k.Value!.ToString()));

        await emailSender.SendConfirmationLinkAsync(user, email, HtmlEncoder.Default.Encode(confirmEmailUrl));
    }
}