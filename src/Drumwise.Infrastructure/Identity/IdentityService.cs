using System.Security.Claims;
using Drumwise.Application.Common.Errors;
using Drumwise.Application.Common.Exceptions;
using Drumwise.Application.Common.Interfaces;
using Drumwise.Application.Common.Models;
using Drumwise.Application.Common.Models.Identity;
using FluentValidation.Results;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http.HttpResults;
using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using ValidationException = FluentValidation.ValidationException;

namespace Drumwise.Infrastructure.Identity;

public class IdentityService(UserManager<ApplicationUser> userManager,
        IUserClaimsPrincipalFactory<ApplicationUser> userClaimsPrincipalFactory,
        IAuthorizationService authorizationService)
    : IIdentityService
{
    public  async Task<string?> GetUserNameAsync(string userId)
    {
        var user = await userManager.Users.FirstAsync(u => u.Id == userId);
        return user.UserName;
    }

    public async Task<bool> IsInRoleAsync(string userId, string role)
    {
        var user = userManager.Users.SingleOrDefault(u => u.Id == userId);
        return user != null && await userManager.IsInRoleAsync(user, role);
    }

    public async Task<bool> AuthorizeAsync(string userId, string policy)
    {
        var user = userManager.Users.SingleOrDefault(u => u.Id == userId);
        if (user is null)
            return false;

        var principal = await userClaimsPrincipalFactory.CreateAsync(user);
        var result = await authorizationService
            .AuthorizeAsync(principal, policy);

        return result.Succeeded;
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
            return Result.Failure(IdentityErrors.UpdatingUserError(result.Errors), ResultType.BadRequest);
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
}